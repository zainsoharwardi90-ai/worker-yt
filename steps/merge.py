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
    """Build a chained atempo filter for per-segment duration fitting in
    synthesize.py (NOT for global video-length matching).

    ratio = tts_duration / target_window; the filter speeds the TTS up by 1/ratio
    so its duration becomes target_window. atempo supports only 0.5-2.0 per
    instance, so out-of-range factors are decomposed by chaining atempo=2.0
    (speed up) or atempo=0.5 (slow down) instances. merge() itself never
    stretches the timeline — it only muxes the already-synchronized track.
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
    """Mux the timestamp-aligned dubbed audio track back onto the video.

    The audio track from synthesize.build_dubbed_audio is a silent base
    exactly as long as the video, with each dubbed segment overlaid at its
    original start timestamp (see audio_assembler.py in the reference
    implementation for the same concept). The timeline must therefore NOT be
    stretched here: a global atempo would shift every segment off its
    timestamp and destroy both lipsync and the silent gaps between dialogue.

    We only cap the output to the video's own duration so the video timeline
    always stays the fixed reference; any last-millisecond audio overrun is
    trimmed and any shortfall leaves trailing silence.
    """
    video_duration = get_duration(video_path)
    audio_duration = get_duration(audio_path)

    if video_duration <= 0:
        raise RuntimeError("Invalid duration from ffprobe (video)")
    if audio_duration <= 0:
        raise RuntimeError("Invalid duration from ffprobe (audio)")

    delta = abs(video_duration - audio_duration)
    if delta > 1.0:
        print(
            f"[WARN] Audio track is {delta:.2f}s off the video timeline "
            f"(video={video_duration:.2f}s, audio={audio_duration:.2f}s). "
            f"Keeping the video timeline as-is; no re-stretching applied."
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
        '-t', f'{video_duration:.6f}',
        '-movflags', '+faststart',
        output_path,
    ]

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
