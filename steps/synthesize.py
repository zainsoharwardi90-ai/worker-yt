import os
import subprocess
import tempfile
from pydub import AudioSegment
from steps.merge import get_duration, build_atempo

TARGET_SAMPLE_RATE = 44100
MAX_SPEED_UP = 1.5


def fit_audio_to_window(tts_path, start, end, tmpdir, idx):
    """Time-stretch a TTS clip so it fits inside the dialogue window [start, end].

    The video timeline is never changed; only the clip is sped up (never slowed
    down). If the clip already fits, it is used unchanged. If it must be sped up
    more than MAX_SPEED_UP, the speed is clamped and the tail is allowed to
    overflow the window rather than destroying audio quality.
    """
    tts_duration = get_duration(tts_path)
    window = max(end - start, 0.2)
    speed = tts_duration / window

    if speed <= 1.0:
        print(
            f"[INFO] Segment {idx}: TTS {tts_duration:.2f}s fits window "
            f"{window:.2f}s; no speed adjustment"
        )
        return tts_path

    if speed > MAX_SPEED_UP:
        print(
            f"[WARN] Segment {idx}: needs {speed:.2f}x (window {window:.2f}s vs "
            f"TTS {tts_duration:.2f}s) which exceeds {MAX_SPEED_UP:.2f}x max. "
            f"Clamping speed; audio may overflow the window."
        )
        speed = MAX_SPEED_UP
    else:
        print(
            f"[INFO] Segment {idx}: fitting TTS {tts_duration:.2f}s into window "
            f"{window:.2f}s at {speed:.2f}x"
        )

    atempo = build_atempo(1.0 / speed)
    fit_path = os.path.join(tmpdir, f"fitted_tts_{idx}.wav")
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


def build_dubbed_audio(video_path, segments, output_audio_path):
    """Build a single combined audio track for the whole video.

    The track starts as pure silence with exactly the video's duration. Each
    dubbed segment is speed-adjusted to fit its original dialogue window and
    then overlaid at its original start timestamp, so silent gaps stay silent.

    segments: list of dicts with keys start, end, tts_path.
    """
    video_duration_ms = int(round(get_duration(video_path) * 1000))
    print(f"[INFO] Creating silent base track of {video_duration_ms}ms")

    base = AudioSegment.silent(duration=video_duration_ms, frame_rate=TARGET_SAMPLE_RATE)

    if not segments:
        print("[WARN] No speech segments found; exporting silent track")
        base.export(output_audio_path, format="wav")
        return output_audio_path

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, seg in enumerate(segments):
            start_ms = int(round(seg["start"] * 1000))
            print(
                f"[INFO] Placing segment {idx + 1}/{len(segments)} at "
                f"{start_ms}ms ({seg['start']:.2f}s)"
            )
            fit_path = fit_audio_to_window(
                seg["tts_path"], seg["start"], seg["end"], tmpdir, idx
            )
            clip = (
                AudioSegment.from_file(fit_path)
                .set_frame_rate(TARGET_SAMPLE_RATE)
                .set_channels(1)
                .set_sample_width(2)
            )
            base = base.overlay(clip, position=start_ms)

        # Clamp to the exact video duration so a segment that overflowed its
        # window past the end of the video can never extend the output.
        base = base[:video_duration_ms]

    print(f"[INFO] Exporting combined audio track: {output_audio_path}")
    base.export(output_audio_path, format="wav")
    return output_audio_path