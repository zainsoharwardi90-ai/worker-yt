import os
import subprocess
import tempfile
from pydub import AudioSegment
from steps.merge import get_duration, build_atempo

TARGET_SAMPLE_RATE = 44100

# --- Configurable synchronization constants (overridable via env vars) ---
# Never slow TTS down below natural speed. A shorter voice keeps its natural
# pace; the remainder of the window stays silent (Case A).
MIN_TEMPO = float(os.environ.get("TTS_MIN_TEMPO", "1.0"))
# Safe maximum speed-up for pitch-preserving atempo (Case B fits inside it;
# Case C uses it as a cap before truncation applies).
MAX_TEMPO = float(os.environ.get("TTS_MAX_TEMPO", "1.5"))
MAX_SPEED_UP = MAX_TEMPO  # legacy alias
# Minimum effective speech window to guard degenerate / fully-overlapped
# segments so a tiny or zero-length window can never divide-by-zero.
MIN_WINDOW_SEC = float(os.environ.get("TTS_MIN_WINDOW_SEC", "0.2"))
# Durations within this many seconds (relative to window length) of the window
# are left at natural speed instead of being stretched. Any residual overflow
# past the window is still caught by the hard-truncation safety net below.
FIT_EPSILON = float(os.environ.get("TTS_FIT_EPSILON", "0.03"))
# Short fade-out applied ONLY at a hard truncation point (the audio was cut to
# fit the window); prevents an audible click at the cut. Within 50-100ms.
TRUNCATE_FADE_MS = int(
    os.environ.get("TTS_TRUNCATE_FADE_MS",
                   os.environ.get("TTS_SAFETY_FADE_MS", "75"))
)

# Per-run truncation accounting (reset each build_dubbed_audio call) so callers
# and tests can see how often the hard safety net fired and by how much.
TRUNCATION_LOG = []
_truncation_cut_ms = 0


def _time_stretch(tts_path, speed, fit_path, idx):
    """Speed up a TTS clip by `speed` (>= 1) with pitch-preserving atempo.

    Returns the fitted path (PCM wav at TARGET_SAMPLE_RATE, mono, 16-bit).
    atempo changes tempo only — pitch and voice quality are preserved — so a
    sentence can be shortened without sounding chipmunk-like.
    """
    atempo = build_atempo(1.0 / speed)
    cmd = [
        'ffmpeg', '-y',
        '-i', tts_path,
        '-af', atempo,
        '-ar', str(TARGET_SAMPLE_RATE), '-ac', '1',
        fit_path,
    ]
    print(f"[INFO] Segment {idx}: running {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        last_line = stderr.splitlines()[-1] if stderr else "no stderr output"
        raise RuntimeError(f"ffmpeg atempo failed for segment {idx}: {last_line}")
    return fit_path


def _load_clip(path):
    """Decode a TTS file into a 44.1kHz mono 16-bit AudioSegment (no click-prone mp3 gaps)."""
    return (
        AudioSegment.from_file(path)
        .set_frame_rate(TARGET_SAMPLE_RATE)
        .set_channels(1)
        .set_sample_width(2)
    )


def _validate_inputs(video_duration, segments):
    """Validate the video timeline and segment list before synthesis.

    Clamps fixable issues (negative starts, ends beyond the video, unsorted
    order) and drops segments whose timings are unusable. Mutates the segment
    dicts in place, adding "_drop" for unusable ones; returns None.
    """
    if not video_duration or video_duration <= 0:
        raise RuntimeError(f"Invalid video duration from ffprobe: {video_duration}")
    print(
        f"[INFO] Validated video duration: {video_duration:.2f}s, "
        f"{len(segments)} segment(s)"
    )
    n_neg = n_over = n_unsorted = 0
    prev_start = -1.0
    for i, seg in enumerate(segments):
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        if start < 0:
            n_neg += 1
            seg["start"] = 0.0
            start = 0.0
        if end <= start:
            print(f"[WARN] Segment {i + 1}: end {end:.2f} <= start {start:.2f}; dropping")
            seg["_drop"] = True
            continue
        if end > video_duration:
            n_over += 1
            seg["end"] = float(video_duration)
            end = float(video_duration)
        if start < prev_start - 0.01:
            n_unsorted += 1
        prev_start = start
    if n_neg:
        print(f"[WARN] {n_neg} segment(s) with negative start clamped to 0")
    if n_over:
        print(f"[WARN] {n_over} segment(s) ending beyond video duration clamped to {video_duration:.2f}s")
    if n_unsorted:
        print("[WARN] Segments were out of order; sorting chronologically before placement")
    segments.sort(key=lambda s: float(s.get("start") or 0.0))


def fit_audio_to_window(tts_path, window_sec, idx, tmpdir):
    """Fit one TTS clip to EXACTLY its allotted window [start, start+window_sec].

    Decision tree (window = the segment's own speech window, always capped at
    the next segment's start / the video end, so it never collides with the
    following voice):

      * TTS fits at natural speed          -> keep natural (Case A / Case B)
      * TTS needs <= MAX_TEMPO speed-up    -> atempo so the WHOLE sentence
                                              lands exactly at window end
      * TTS too long even at MAX_TEMPO     -> max atempo first, then HARD
                                              TRUNCATE the leftover past the
                                              window end, with a short fade-out

    A segment's audio NEVER extends past its allowed window: truncation with a
    fade-out is the unconditional safety net, regardless of the cap value.
    If truncation cuts largely, raising MAX_TEMPO (e.g. 1.6-1.7x) reduces how
    often it fires — but the net stays in place.

    Returns (clip, action_description, truncated_ms).
    """
    window_ms = int(round(window_sec * 1000))
    if window_ms <= 0:
        window_ms = int(round(MIN_WINDOW_SEC * 1000))
    tts_duration = get_duration(tts_path)

    required_speed = tts_duration / (window_ms / 1000.0) if window_ms > 0 else float("inf")

    if required_speed <= MIN_TEMPO + FIT_EPSILON:
        # Case A: fits at natural speed (or trivially over; the net below
        # clips any residual overflow past the window).
        clip = _load_clip(tts_path)
        action = "natural speed"
    elif required_speed <= MAX_TEMPO:
        # Case B: exactly fit the full sentence into the window.
        fit_path = os.path.join(tmpdir, f"fitted_tts_{idx}.wav")
        _time_stretch(tts_path, required_speed, fit_path, idx + 1)
        clip = _load_clip(fit_path)
        action = f"time-stretched x{required_speed:.2f}"
    else:
        # Case C: apply max-speed stretch first (keeps as much of the sentence
        # as possible), then hard-truncate what still does not fit.
        fit_path = os.path.join(tmpdir, f"fitted_tts_{idx}.wav")
        _time_stretch(tts_path, MAX_TEMPO, fit_path, idx + 1)
        clip = _load_clip(fit_path)
        action = f"max tempo x{MAX_TEMPO:.2f}"

    # ---- HARD SAFETY NET: never exceed the window -------------------------
    truncated_ms = 0
    if len(clip) > window_ms:
        truncated_ms = len(clip) - window_ms
        fade = min(TRUNCATE_FADE_MS, len(clip) // 4)
        clip = clip[:window_ms].fade_out(fade)
        action += f", HARD-TRUNCATED {truncated_ms}ms"
        print(
            f"[TRUNCATE] Segment {idx + 1}: cut {truncated_ms}ms to fit "
            f"{window_ms}ms window ({tts_duration:.2f}s TTS at "
            f"{required_speed:.2f}x speed -> clipped with {fade}ms fade-out)"
        )
        TRUNCATION_LOG.append({"idx": idx + 1, "cut_ms": truncated_ms})

    return clip, action, truncated_ms


def build_dubbed_audio(video_path, segments, output_audio_path):
    """Build a single timeline-based combined audio track for the whole video.

    The track starts as pure silence with exactly the video's duration; audio is
    ONLY placed where a valid speech segment exists. Per segment:

      1. The available speech window is the segment's own [start, end], clamped
         so it never crosses the next segment's start (overlap guard) or the
         end of the video.
      2. The measured TTS duration is compared with the window:
         - shorter           -> kept at natural speed, silence fills the rest
         - slightly longer   -> pitch-preserving atempo so the WHOLE sentence
                                fits exactly inside the window
         - much longer       -> max-speed atempo first; whatever STILL does not
                                fit is hard-truncated at the window end with a
                                short fade-out. The clip NEVER crosses the
                                window boundary — silence/music gaps between
                                segments are never invaded.
      3. Each clip is overlaid at the original start timestamp.
      4. The final track is clamped to the exact video duration.

    segments: list of dicts with keys start, end, tts_path.
    """
    global _truncation_cut_ms
    del TRUNCATION_LOG[:]
    _truncation_cut_ms = 0

    video_duration = get_duration(video_path)
    _validate_inputs(video_duration, segments)
    video_duration_ms = int(round(video_duration * 1000))
    print(
        f"[INFO] Creating silent base track of {video_duration_ms}ms "
        f"(video {video_duration:.2f}s)"
    )

    base = AudioSegment.silent(duration=video_duration_ms, frame_rate=TARGET_SAMPLE_RATE)

    ordered = sorted(
        [s for s in segments if s.get("tts_path") and not s.get("_drop")],
        key=lambda s: s["start"],
    )
    if not ordered:
        print("[WARN] No valid speech segments found; exporting silent track")
        base.export(output_audio_path, format="wav")
        return output_audio_path

    placed_actions = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, seg in enumerate(ordered):
            start = float(seg["start"])
            end = float(seg["end"])
            next_start = (
                float(ordered[idx + 1]["start"])
                if idx + 1 < len(ordered)
                else video_duration
            )

            # ---- window boundaries ---------------------------------------
            # Primary window: the segment's OWN [start, end], never past the
            # next segment's start (overlap guard) nor the end of the video.
            primary_end = min(end, next_start, video_duration)
            if primary_end <= start:
                primary_end = start + MIN_WINDOW_SEC
            window = primary_end - start
            start_ms = int(round(start * 1000))

            tts_path = seg["tts_path"]
            if not os.path.isfile(tts_path):
                print(f"[WARN] Segment {idx + 1}: TTS file missing: {tts_path}; skipping")
                continue
            tts_duration = get_duration(tts_path)
            if tts_duration <= 0:
                print(f"[WARN] Segment {idx + 1}: could not measure TTS duration; skipping")
                continue

            overlap = next_start < end
            required_speed = tts_duration / window
            if overlap:
                overlap_status = (
                    f"OVERLAP -> window capped to {window:.2f}s "
                    f"(next boundary at {next_start:.2f}s)"
                )
            else:
                overlap_status = f"no overlap (next boundary at {next_start:.2f}s)"
            print(f"\nSegment {idx + 1}")
            print(f"Original: {start:.2f}s -> {end:.2f}s")
            print(f"Original duration: {max(end - start, 0):.2f}s")
            print(f"Available duration: {window:.2f}s")
            print(f"TTS duration: {tts_duration:.2f}s")
            print(f"Required tempo: {required_speed:.2f}x")
            print(f"Overlap status: {overlap_status}")

            clip, action, truncated_ms = fit_audio_to_window(
                tts_path, window, idx, tmpdir
            )
            final_duration = len(clip) / 1000.0
            _truncation_cut_ms += truncated_ms

            print(f"Final duration: {final_duration:.2f}s")
            print(f"Placement: {start:.2f}s")
            print(f"Action: {action}")
            placed_actions.append(action)

            if len(clip) <= 0:
                print(f"[WARN] Segment {idx + 1}: empty clip after fitting; skipping")
                continue

            # Placement at the original start; the clip has already been
            # capped to the window so it can never collide with the next voice
            # or the video end.
            base = base.overlay(clip, position=start_ms)

        # Clamp to the exact video duration — the timeline never exceeds the video.
        base = base[:video_duration_ms]

    print(f"\n[INFO] Placement summary ({len(ordered)} segments):")
    for act in placed_actions:
        print(f"  - {act}")
    if TRUNCATION_LOG:
        avg = _truncation_cut_ms / len(TRUNCATION_LOG)
        print(
            f"[TRUNCATE] Summary: {len(TRUNCATION_LOG)}/{len(ordered)} segments "
            f"hard-truncated (total {_truncation_cut_ms}ms, avg {avg:.0f}ms per "
            f"truncation). If truncation is frequent, consider raising MAX_TEMPO "
            f"(currently {MAX_TEMPO:.1f}x, e.g. to 1.6-1.7x) to reduce speech loss."
        )
    else:
        print(
            f"[INFO] Truncation summary: 0/{len(ordered)} segments needed "
            f"truncation — all TTS audio fit within its windows."
        )
    print(f"[INFO] Exporting combined audio track: {output_audio_path}")
    base.export(output_audio_path, format="wav")

    # Post-synthesis validation: the timeline must match the video.
    final_duration = get_duration(output_audio_path)
    if abs(final_duration - video_duration) > 0.2:
        print(
            f"[WARN] Post-synthesis check: final audio {final_duration:.2f}s differs "
            f"from video {video_duration:.2f}s (delta {abs(final_duration - video_duration):.2f}s)"
        )
    else:
        print(
            f"[INFO] Post-synthesis check: final audio {final_duration:.2f}s "
            f"matches video timeline {video_duration:.2f}s"
        )
    return output_audio_path
