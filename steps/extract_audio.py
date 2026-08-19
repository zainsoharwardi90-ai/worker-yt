import os
import subprocess


def is_valid_video(video_path):
    """Verify a file exists, is non-empty, and is a real video container.

    Returns (valid: bool, reason: str|None).
    """
    if not os.path.isfile(video_path):
        return False, f"File does not exist: {video_path}"
    if os.path.getsize(video_path) == 0:
        return False, f"File is empty (0 bytes): {video_path}"

    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=format_name',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False, f"ffprobe timed out while inspecting: {video_path}"
    except Exception as e:
        return False, f"ffprobe failed to run: {e}"

    if result.returncode != 0:
        return False, (result.stderr.strip() or f"ffprobe exit code {result.returncode}")
    if not result.stdout.strip():
        return False, "ffprobe could not identify a valid video format"
    return True, None


def extract_audio(input_video_path, output_audio_path, sample_rate=16000):
    valid, reason = is_valid_video(input_video_path)
    if not valid:
        raise RuntimeError(f"Invalid input video: {reason}")

    cmd = [
        'ffmpeg', '-y', '-i', input_video_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', str(sample_rate), '-ac', '1',
        output_audio_path,
    ]

    print(f"[INFO] Running ffmpeg audio extraction: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        print(f"[ERROR] ffmpeg audio extraction failed with exit code {result.returncode}")
        print("[ffmpeg STDOUT]")
        print(stdout if stdout else "(no stdout)")
        print("[ffmpeg STDERR]")
        print(stderr if stderr else "(no stderr)")
        last_line = stderr.splitlines()[-1] if stderr else "no stderr output"
        raise RuntimeError(
            f"ffmpeg audio extraction failed (exit code {result.returncode}): {last_line}"
        )

    print(f"[INFO] Audio extracted successfully: {output_audio_path}")
    return output_audio_path
