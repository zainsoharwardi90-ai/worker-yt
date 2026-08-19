import os
import time
import tempfile
import subprocess
import requests

HF_ENDPOINT = os.environ.get(
    "HF_ENDPOINT",
    "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3",
)
MAX_ATTEMPTS = int(os.environ.get("HF_MAX_ATTEMPTS", "3"))
RETRY_WAIT_SECONDS = 20
CHUNK_THRESHOLD_SEC = 60
CHUNK_DURATION_SEC = 50
REQUEST_TIMEOUT = 300
# Maximum gap (seconds) between two segments that gets merged into one dialogue
# window. Preserves pauses larger than this so meaningful silences survive.
MIN_SEGMENT_GAP = float(os.environ.get("MIN_SEGMENT_GAP", "0.5"))
RATE_LIMIT_DELAY_SECONDS = float(os.environ.get("RATE_LIMIT_DELAY_SECONDS", "1.0"))


def get_duration(audio_path):
    """Return audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            audio_path,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed to get duration for {audio_path}: {result.stderr.strip()}"
        )
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Could not parse duration for {audio_path}: {result.stdout!r}")


def split_audio(audio_path, chunk_duration=CHUNK_DURATION_SEC):
    """Split audio into ~chunk_duration-second wav chunks using ffmpeg."""
    total = get_duration(audio_path)
    chunks = []
    start = 0.0
    idx = 0
    while start < total:
        chunk_path = os.path.join(
            tempfile.gettempdir(), f"tts_chunk_{os.getpid()}_{idx:04d}.wav"
        )
        cmd = [
            'ffmpeg', '-y',
            '-i', audio_path,
            '-ss', f'{start:.3f}',
            '-t', f'{chunk_duration:.3f}',
            '-ac', '1',
            '-ar', '16000',
            chunk_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        chunks.append(chunk_path)
        start += chunk_duration
        idx += 1
    return chunks


def merge_close_segments(segments, max_gap=MIN_SEGMENT_GAP):
    """Merge whisper segments separated by tiny pauses into one dialogue.

    This keeps dialogues (and their start/end windows) intact so a single
    TTS utterance maps to one continuous speech window instead of several
    overlapping micro-clips. Segments that actually overlap (start before the
    previous end) are merged as well — this is the deterministic overlap
    resolution used by finalize_segments.
    """
    if not segments:
        return segments
    merged = []
    current = dict(segments[0])
    for seg in segments[1:]:
        if seg["start"] - current["end"] <= max_gap:
            current["end"] = max(current["end"], seg["end"])
            current["text"] = (current["text"] + " " + seg["text"]).strip()
        else:
            merged.append(current)
            current = dict(seg)
    merged.append(current)
    return merged


def finalize_segments(segments, duration):
    """Sort, merge, clamp and validate raw Whisper segments for the GLOBAL timeline.

    Guarantees for every returned segment:
      * start >= 0 and end > start
      * chronological order (sorted by start)
      * no overlap (overlapping windows are merged deterministically)
      * no timestamp beyond the media duration

    Fixable issues are clamped and logged so timing problems are diagnosable.
    """
    if not segments:
        return []

    raw = []
    for s in segments:
        try:
            start = float(s.get("start") or 0.0)
            end = float(s.get("end") or start)
        except (TypeError, ValueError):
            print(f"[WARN] Segment with non-numeric timestamp: {s!r}; skipping")
            continue
        raw.append({
            "start": start,
            "end": end if end > start else start + 0.1,
            "text": s.get("text") or "",
        })

    neg_clamped = sum(1 for s in raw if s["start"] < 0)
    for s in raw:
        if s["start"] < 0:
            s["start"] = 0.0
            s["end"] = max(s["end"], 0.0)

    raw.sort(key=lambda s: s["start"])
    merged = merge_close_segments(raw)

    fixed = []
    clamped_end = 0
    dropped = 0
    for s in merged:
        start = s["start"]
        end = min(s["end"], duration)
        if end <= start:
            print(f"[WARN] Dropping degenerate segment [{start:.3f}, {end:.3f}]: {s['text'][:60]!r}")
            dropped += 1
            continue
        if s["end"] > duration:
            clamped_end += 1
        fixed.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text": s["text"],
        })

    # Final overlap safety net (merge_close_segments already handles overlaps).
    for prev, cur in zip(fixed, fixed[1:]):
        if cur["start"] < prev["end"]:
            print(
                f"[WARN] Overlap resolved: segment ending at {prev['end']:.3f}s "
                f"capped to next segment start {cur['start']:.3f}s"
            )
            prev["end"] = cur["start"]

    print(
        f"[TRANSCRIBE] {len(fixed)} segments "
        f"(merged from {len(raw)} raw, {neg_clamped} negative-start clamped, "
        f"{clamped_end} end-over-duration clamped, {dropped} dropped)"
    )
    return fixed


def _request_url(with_timestamps):
    if not with_timestamps:
        return HF_ENDPOINT
    param = "return_timestamps=true"
    if "?" in HF_ENDPOINT:
        return f"{HF_ENDPOINT}&{param}"
    return f"{HF_ENDPOINT}?{param}"


def _transcribe_request(audio_bytes, token, chunk_offset=0.0, fallback_duration=CHUNK_DURATION_SEC):
    """Send one chunk/request to the HF API; returns a list of segments.

    Each segment is {"start": float, "end": float, "text": str} with times
    already offset by chunk_offset. Falls back to one segment per request if
    the API does not return per-chunk timestamps.
    """
    last_error = None
    use_timestamps = True
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(
                _request_url(use_timestamps),
                data=audio_bytes,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "audio/wav",
                },
                timeout=REQUEST_TIMEOUT,
            )

            if resp.status_code in (503, 504):
                last_error = f"Model loading / timeout ({resp.status_code}): {resp.text[:200]}"
                print(
                    f"[WARN] Whisper API returned {resp.status_code} on attempt "
                    f"{attempt}/{MAX_ATTEMPTS}. Waiting {RETRY_WAIT_SECONDS}s..."
                )
                time.sleep(RETRY_WAIT_SECONDS)
                continue

            if resp.status_code == 400 and use_timestamps:
                last_error = f"API rejected return_timestamps (400): {resp.text[:200]}"
                print(
                    f"[WARN] API rejected return_timestamps on attempt "
                    f"{attempt}/{MAX_ATTEMPTS}. Retrying without timestamps..."
                )
                use_timestamps = False
                time.sleep(RETRY_WAIT_SECONDS)
                continue

            if resp.status_code != 200:
                raise RuntimeError(
                    f"HF API returned HTTP {resp.status_code}: {resp.text[:500]}"
                )

            data = resp.json()
            segments = []
            for chunk in data.get("chunks") or []:
                timestamp = chunk.get("timestamp") or [0, 0]
                start = (timestamp[0] or 0) + chunk_offset
                end = (timestamp[1] or start) + chunk_offset
                text = (chunk.get("text") or "").strip()
                if text:
                    segments.append({"start": start, "end": end, "text": text})

            if not segments:
                text = (data.get("text") or "").strip()
                if not text:
                    raise RuntimeError(
                        f"HF API returned no transcription: {resp.text[:500]}"
                    )
                segments = [{
                    "start": chunk_offset,
                    "end": chunk_offset + fallback_duration,
                    "text": text,
                }]

            return segments

        except requests.RequestException as e:
            last_error = f"Request failed: {e}"
            print(f"[WARN] Transcription attempt {attempt}/{MAX_ATTEMPTS} failed: {e}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_WAIT_SECONDS)

    raise RuntimeError(f"Transcription failed after {MAX_ATTEMPTS} attempts: {last_error}")


def transcribe(audio_path, source_lang=None):
    """Transcribe audio and return dialogue segments with timestamps.

    Returns a sorted list of {"start": float, "end": float, "text": str}.
    Long audio (>60s) is split into ~50s chunks; chunk-local timestamps are
    offset so all returned times are relative to the original file.
    """
    token = os.environ.get("HF_API_TOKEN")
    if not token:
        raise RuntimeError("HF_API_TOKEN environment variable is not set")

    duration = get_duration(audio_path)
    all_segments = []

    if duration > CHUNK_THRESHOLD_SEC:
        print(
            f"[INFO] Audio duration {duration:.2f}s > {CHUNK_THRESHOLD_SEC}s. "
            f"Splitting into chunks..."
        )
        chunks = split_audio(audio_path)
        try:
            for i, chunk_path in enumerate(chunks):
                with open(chunk_path, "rb") as f:
                    audio_bytes = f.read()
                # GLOBAL timeline offset: chunk i covers [i*CHUNK_DURATION_SEC, ...).
                # chunk-local Whisper timestamps are shifted by this offset so every
                # segment lands on the original audio/video timeline.
                chunk_offset = i * CHUNK_DURATION_SEC
                remaining = max(duration - chunk_offset, 0.0)
                chunk_len = min(CHUNK_DURATION_SEC, remaining)
                print(
                    f"[INFO] Transcribing chunk {i + 1}/{len(chunks)} "
                    f"({chunk_offset:.1f}s-{chunk_offset + chunk_len:.1f}s): {chunk_path}"
                )
                all_segments.extend(
                    _transcribe_request(
                        audio_bytes, token,
                        chunk_offset=chunk_offset,
                        fallback_duration=max(chunk_len, 0.1),
                    )
                )
                if i < len(chunks) - 1:
                    print(
                        f"[INFO] Rate-limit pause of {RATE_LIMIT_DELAY_SECONDS}s "
                        f"between chunks"
                    )
                    time.sleep(RATE_LIMIT_DELAY_SECONDS)
        finally:
            for chunk_path in chunks:
                try:
                    os.remove(chunk_path)
                    print(f"[INFO] Cleaned up {chunk_path}")
                except OSError as e:
                    print(f"[WARN] Failed to remove {chunk_path}: {e}")
    else:
        print(f"[INFO] Audio duration {duration:.2f}s <= {CHUNK_THRESHOLD_SEC}s. "
              f"Transcribing directly.")
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        all_segments = _transcribe_request(
            audio_bytes, token, chunk_offset=0.0, fallback_duration=duration
        )

    return finalize_segments(all_segments, duration)
