import os
import tempfile
import cloudinary
import cloudinary.uploader
import requests
import psycopg2
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from steps.extract_audio import extract_audio, is_valid_video
from steps.transcribe import transcribe
from steps.translate import translate
from steps.text_to_speech import generate_speech
from steps.synthesize import build_dubbed_audio
from steps.merge import merge, get_duration

NEON_DATABASE_URL = os.environ.get("NEON_DATABASE_URL", "")

cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
)

app = FastAPI(title="yt-dubber Worker")


class ProcessRequest(BaseModel):
    job_id: str
    video_url: str
    source_lang: str
    target_lang: str


def download_file(url, dest_path):
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


def upload_to_cloudinary(key, file_path):
    result = cloudinary.uploader.upload(
        file_path,
        resource_type="video",
        public_id=key.rsplit(".", 1)[0],
    )
    return result["secure_url"]


def update_job_status(job_id, status, output_url=None, error=None):
    if not NEON_DATABASE_URL:
        return
    try:
        conn = psycopg2.connect(NEON_DATABASE_URL)
        cur = conn.cursor()
        fields = ["updated_at = NOW()"]
        params = []
        if status:
            fields.append("status = %s")
            params.append(status)
        if output_url:
            fields.append("output_video_url = %s")
            params.append(output_url)
        if error:
            fields.append("error_message = %s")
            params.append(error)
        params.append(job_id)
        cur.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = %s", params)
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[ERROR] DB update failed for job {job_id}: {e}")


def process_job(job_data):
    job_id = job_data["jobId"]
    input_url = job_data["inputVideoUrl"]
    source_lang = job_data.get("sourceLang")
    target_lang = job_data["targetLang"]

    print(f"[INFO] Processing job {job_id}: {source_lang} -> {target_lang}")
    update_job_status(job_id, "processing")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            video_path = os.path.join(tmpdir, "input_video.mp4")
            audio_path = os.path.join(tmpdir, "extracted_audio.wav")
            dubbed_audio_path = os.path.join(tmpdir, "combined_audio.wav")
            output_video_path = os.path.join(tmpdir, "dubbed_output.mp4")

            print(f"[INFO] Downloading input video for job {job_id}")
            download_file(input_url, video_path)

            valid, reason = is_valid_video(video_path)
            if not valid:
                raise RuntimeError(f"Downloaded video is invalid: {reason}")

            video_duration = get_duration(video_path)
            if video_duration <= 0:
                raise RuntimeError(f"Invalid video duration: {video_duration}")
            print(f"[INFO] Video duration for job {job_id}: {video_duration:.2f}s")

            print(f"[INFO] Extracting audio for job {job_id}")
            extract_audio(video_path, audio_path)

            print(f"[INFO] Transcribing for job {job_id}")
            segments = transcribe(audio_path, source_lang)

            total = len(segments)
            kept = 0
            for idx, seg in enumerate(segments):
                seg_start = seg["start"]
                seg_end = seg["end"]
                window = max(seg_end - seg_start, 0.2)
                print(
                    f"[INFO] Translating segment {idx + 1}/{total} "
                    f"({seg_start:.2f}s-{seg_end:.2f}s, window {window:.2f}s)"
                )
                try:
                    translated_text = translate(seg["text"], target_lang)
                except Exception as e:
                    print(
                        f"[WARN] Segment {idx + 1}: translation failed ({e}); "
                        f"leaving silence"
                    )
                    seg["tts_path"] = None
                    continue
                if not (translated_text or "").strip():
                    print(
                        f"[WARN] Segment {idx + 1}: empty translation; leaving silence"
                    )
                    seg["tts_path"] = None
                    continue
                seg["translated_text"] = translated_text

                tts_path = os.path.join(tmpdir, f"speech_{idx}.mp3")
                print(f"[INFO] Generating speech for segment {idx + 1}/{total}")
                try:
                    generate_speech(translated_text, tts_path, target_lang)
                except Exception as e:
                    print(
                        f"[WARN] Segment {idx + 1}: TTS generation failed ({e}); "
                        f"leaving silence"
                    )
                    seg["tts_path"] = None
                    continue
                if not os.path.isfile(tts_path) or os.path.getsize(tts_path) == 0:
                    print(
                        f"[WARN] Segment {idx + 1}: TTS produced no audio; "
                        f"leaving silence"
                    )
                    seg["tts_path"] = None
                    continue

                tts_duration = get_duration(tts_path)
                seg["tts_duration"] = tts_duration
                print(
                    f"[INFO] Segment {idx + 1}: original {seg_start:.2f}s->{seg_end:.2f}s "
                    f"(window {window:.2f}s), TTS {tts_duration:.2f}s, "
                    f"required tempo {tts_duration / window:.2f}x"
                )
                seg["tts_path"] = tts_path
                kept += 1

            skipped = [s for s in segments if not s.get("tts_path")]
            segments = [s for s in segments if s.get("tts_path")]
            print(
                f"[INFO] {kept}/{total} segments have TTS "
                f"({len(skipped)} left silent; positions preserved by timestamps)"
            )

            print(f"[INFO] Building combined dubbed audio track for job {job_id}")
            build_dubbed_audio(video_path, segments, dubbed_audio_path)

            dubbed_duration = get_duration(dubbed_audio_path)
            print(
                f"[INFO] Dubbed audio duration {dubbed_duration:.2f}s vs "
                f"video {video_duration:.2f}s"
            )

            print(f"[INFO] Muxing audio with original video for job {job_id}")
            merge(video_path, dubbed_audio_path, output_video_path)

            cloudinary_key = f"output/{job_id}-dubbed.mp4"
            print(f"[INFO] Uploading result to Cloudinary for job {job_id}")
            output_url = upload_to_cloudinary(cloudinary_key, output_video_path)

            update_job_status(job_id, "done", output_url=output_url)
            print(f"[INFO] Job {job_id} completed successfully")

        except Exception as e:
            print(f"[ERROR] Job {job_id} failed: {e}")
            update_job_status(job_id, "failed", error=str(e))


def run_process_job(req: ProcessRequest):
    payload = {
        "jobId": req.job_id,
        "inputVideoUrl": req.video_url,
        "sourceLang": req.source_lang,
        "targetLang": req.target_lang,
    }
    try:
        process_job(payload)
    except Exception as e:
        print(f"[ERROR] Unhandled exception for job {req.job_id}: {e}")
        update_job_status(req.job_id, "failed", error=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process")
def process(req: ProcessRequest, background_tasks: BackgroundTasks):
    print(f"[INFO] Accepted job {req.job_id} for processing")
    background_tasks.add_task(run_process_job, req)
    return {"job_id": req.job_id, "status": "accepted"}
