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
# Case C uses it as a cap before a deterministic fallback applies).
MAX_TEMPO = float(os.environ.get("TTS_MAX_TEMPO", "1.5"))
MAX_SPEED_UP = MAX_TEMPO  # legacy alias
# Minimum effective speech window to guard degenerate / fully-overlapped
# segments so a tiny or zero-length window can never divide-by-zero.
MIN_WINDOW_SEC = float(os.environ.get("TTS_MIN_WINDOW_SEC", "0.2"))
# Durations within this many seconds of the window are left at natural speed
# instead of being stretched (avoids pointless micro atempo on trivial diffs).
FIT_EPSILON = float(os.environ.get("TTS_FIT_EPSILON", "0.03"))
# Short fade-out applied only when the *absolute* safety boundary cuts a clip,
# preventing an audible click at the cut point.
SAFETY_FADE_MS = int(os.environ.get("TTS_SAFETY_FADE_MS", "40"))


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


def build_dubbed_audio(video_path, segments, output_audio_path):
    """Build a single timeline-based combined audio track for the whole video.

    The track starts as pure silence with exactly the video's duration; audio is
    ONLY placed where a valid speech segment exists. Per segment:

      1. The available speech window is the segment's own [start, end], clamped
         so it never crosses the next segment's start (deterministic overlap
         guard) or the end of the video.
      2. The measured TTS duration is compared with the window:
         - shorter           -> kept at natural speed, silence fills the rest
         - slightly longer   -> pitch-preserving atempo so the WHOLE sentence
                                fits exactly inside the window
         - much longer       -> max-speed atempo first; if it still overflows it
                                may extend into the gap up to the next segment's
                                start / video end (never overlapping the next
                                voice). Only the absolute overflow beyond that
                                hard boundary is faded + trimmed, as a last
                                resort, and logged loudly.
      3. Each clip is overlaid at the original start timestamp.
      4. The final track is clamped to the exact video duration.

    segments: list of dicts with keys start, end, tts_path.
    """
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

            # ---- boundaries -------------------------------------------------
            # Primary window: the segment's own [start, end], never past the
            # next segment's start (overlap guard) nor the end of the video.
            primary_end = min(end, next_start, video_duration)
            if primary_end <= start:
                primary_end = start + MIN_WINDOW_SEC
            window = primary_end - start

            # Absolute boundary: this clip must NEVER pass the next segment's
            # start (would mix two voices) or the end of the video.
            hard_end = min(next_start, video_duration)
            hard_end_ms = int(round(hard_end * 1000))
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

            if required_speed <= MIN_TEMPO + FIT_EPSILON:
                # Case A: TTS fits at natural speed. No stretching, silence
                # for the remaining part of the window.
                clip = _load_clip(tts_path)
                final_duration = len(clip) / 1000.0
                action = (
                    f"natural speed + remaining silence "
                    f"({max(window - final_duration, 0):.2f}s)"
                )
                print(f"Final duration: {final_duration:.2f}s")

            elif required_speed <= MAX_TEMPO:
                # Case B: slightly longer — stretch the complete sentence so it
                # lands exactly at the end of the window.
                fit_path = os.path.join(tmpdir, f"fitted_tts_{idx}.wav")
                _time_stretch(tts_path, required_speed, fit_path, idx + 1)
                clip = _load_clip(fit_path)
                final_duration = get_duration(fit_path)
                action = "time-stretched"
                print(f"Final duration: {final_duration:.2f}s")

            else:
                # Case C: significantly longer. First apply max-speed stretch so
                # the whole sentence survives, then decide if more room is needed.
                print(
                    f"[WARN] Required tempo {required_speed:.2f}x exceeds "
                    f"max {MAX_TEMPO:.2f}x"
                )
                fit_path = os.path.join(tmpdir, f"fitted_tts_{idx}.wav")
                _time_stretch(tts_path, MAX_TEMPO, fit_path, idx + 1)
                clip = _load_clip(fit_path)
                final_duration = get_duration(fit_path)

                if start + final_duration <= hard_end + FIT_EPSILON:
                    action = (
                        f"fallback: max tempo stretch, ends at "
                        f"{start + final_duration:.2f}s (extends "
                        f"{start + final_duration - end:.2f}s past segment end, "
                        f"before next segment)"
                    )
                    print(f"Final duration: {final_duration:.2f}s")
                else:
                    action = "fallback: max tempo + safety trim at hard boundary"
                    print(
                        f"Final duration (before safety trim): {final_duration:.2f}s, "
                        f"will be capped at {max(hard_end - start, 0):.2f}s"
                    )
                    print(
                        f"WARNING: TTS significantly exceeds available speech window"
                    )

            print(f"Placement: {start:.2f}s")
            print(f"Action: {action}")
            placed_actions.append(action)

            # ---- placement with absolute safety boundary --------------------
            max_len_ms = hard_end_ms - start_ms
            if max_len_ms <= 0:
                print(
                    f"[WARN] Segment {idx + 1}: no room on the timeline; "
                    f"skipping to avoid overlap"
                )
                continue
            if len(clip) > max_len_ms:
                overflow_ms = len(clip) - max_len_ms
                fade = min(SAFETY_FADE_MS, len(clip) // 4)
                clip = clip[:max_len_ms].fade_out(fade)
                print(
                    f"[WARN] Segment {idx + 1}: faded + trimmed {overflow_ms}ms of "
                    f"audio past the absolute boundary at {hard_end:.2f}s to protect "
                    f"the next segment / video end"
                )
            base = base.overlay(clip, position=start_ms)

        # Clamp to the exact video duration — the timeline never exceeds the video.
        base = base[:video_duration_ms]

    print(f"\n[INFO] Placement summary ({len(ordered)} segments):")
    for act in placed_actions:
        print(f"  - {act}")
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
