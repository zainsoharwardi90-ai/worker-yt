import subprocess


def get_duration(path):
    """Return media duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            path,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed to get duration for {path}: {result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Could not parse duration for {path}: {result.stdout!r}")


def build_atempo(ratio):
    """Build a chained atempo filter to match audio duration to the video.

    ratio = video_duration / tts_audio_duration. To make the TTS audio last
    as long as the video we need a speed factor of 1/ratio. atempo supports
    only 0.5-2.0 per instance, so out-of-range factors are decomposed by
    chaining atempo=2.0 (speed up) or atempo=0.5 (slow down) instances.
    """
    speed = 1.0 / ratio
    filters = []
    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        filters.append("atempo=0.5")
        speed /= 0.5
    filters.append(f"atempo={speed:.6f}")
    return ",".join(filters)


def merge(video_path, audio_path, output_path):
    video_duration = get_duration(video_path)
    audio_duration = get_duration(audio_path)

    if video_duration <= 0 or audio_duration <= 0:
        raise RuntimeError("Invalid duration from ffprobe (video or audio)")

    ratio = video_duration / audio_duration

    atempo_filter = None
    if ratio < 0.95 or ratio > 1.05:
        atempo_filter = build_atempo(ratio)
        print(
            f"[INFO] Duration mismatch: video={video_duration:.2f}s, "
            f"tts={audio_duration:.2f}s, ratio={ratio:.3f}. "
            f"Applying audio filter: {atempo_filter}"
        )
    else:
        print(
            f"[INFO] Durations close enough: video={video_duration:.2f}s, "
            f"tts={audio_duration:.2f}s, ratio={ratio:.3f}. No speed adjustment."
        )

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-map', '0:v:0',
        '-map', '1:a:0',
    ]
    if atempo_filter:
        cmd += ['-filter:a', atempo_filter]
    cmd += ['-shortest', '-movflags', '+faststart', output_path]

    print(f"[INFO] Running merge: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        print(f"[ERROR] ffmpeg merge failed with exit code {result.returncode}")
        print("[ffmpeg STDERR]")
        print(stderr if stderr else "(no stderr)")
        last_line = stderr.splitlines()[-1] if stderr else "no stderr output"
        raise RuntimeError(f"ffmpeg merge failed (exit code {result.returncode}): {last_line}")

    print(f"[INFO] Merge completed: {output_path}")
    return output_path
