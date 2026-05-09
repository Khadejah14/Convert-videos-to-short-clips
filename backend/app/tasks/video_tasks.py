import os
import uuid
import traceback
from datetime import datetime
from celery import chain, group, chord
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.tasks.celery_app import celery_app
from app.core.config import get_settings
from app.models.job import Job, Clip, JobStatus, ClipStatus

settings = get_settings()

sync_db_url = settings.DATABASE_URL.replace("+asyncpg", "")
sync_engine = create_engine(sync_db_url, pool_size=5, max_overflow=10)
SyncSession = sessionmaker(bind=sync_engine)


def get_sync_db() -> Session:
    return SyncSession()


def update_job_status(
    db: Session,
    job_id: uuid.UUID,
    status: JobStatus,
    progress: float = None,
    current_step: str = None,
    error_message: str = None,
    celery_task_id: str = None,
):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return

    job.status = status
    if progress is not None:
        job.progress = progress
    if current_step is not None:
        job.current_step = current_step
    if error_message is not None:
        job.error_message = error_message
    if celery_task_id is not None:
        job.celery_task_id = celery_task_id

    if status == JobStatus.COMPLETED:
        job.completed_at = datetime.utcnow()
        job.progress = 100.0

    db.commit()


def update_clip_status(
    db: Session,
    clip_id: uuid.UUID,
    status: ClipStatus,
    original_path: str = None,
    cropped_path: str = None,
    final_path: str = None,
    error_message: str = None,
    no_captions: bool = None,
):
    clip = db.query(Clip).filter(Clip.id == clip_id).first()
    if not clip:
        return

    clip.status = status
    if original_path is not None:
        clip.original_path = original_path
    if cropped_path is not None:
        clip.cropped_path = cropped_path
    if final_path is not None:
        clip.final_path = final_path
    if error_message is not None:
        clip.error_message = error_message
    if no_captions is not None:
        clip.no_captions = no_captions

    db.commit()


@celery_app.task(
    bind=True,
    name="app.tasks.video_tasks.extract_audio_task",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def extract_audio_task(self, job_id: str) -> str:
    db = get_sync_db()
    try:
        job_uuid = uuid.UUID(job_id)
        update_job_status(
            db, job_uuid, JobStatus.EXTRACTING_AUDIO,
            progress=10.0,
            current_step="Extracting audio...",
            celery_task_id=self.request.id,
        )

        from moviepy.editor import VideoFileClip
        import tempfile

        job = db.query(Job).filter(Job.id == job_uuid).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")

        video = VideoFileClip(job.video_path)
        audio_path = os.path.join(settings.TEMP_DIR, f"{job_id}_audio.mp3")
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)

        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()

        job.audio_path = audio_path
        db.commit()

        return audio_path

    except Exception as exc:
        update_job_status(
            db, uuid.UUID(job_id), JobStatus.FAILED,
            error_message=f"Audio extraction failed: {str(exc)}",
        )
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(
    bind=True,
    name="app.tasks.video_tasks.transcribe_audio_task",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def transcribe_audio_task(self, job_id: str, audio_path: str) -> str:
    db = get_sync_db()
    try:
        job_uuid = uuid.UUID(job_id)
        update_job_status(
            db, job_uuid, JobStatus.TRANSCRIBING,
            progress=25.0,
            current_step="Transcribing audio with Whisper...",
        )

        import openai

        openai.api_key = settings.OPENAI_API_KEY

        job = db.query(Job).filter(Job.id == job_uuid).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")

        with open(audio_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        transcript_text = ""
        for segment in transcript.segments:
            transcript_text += f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}\n"

        job.transcript_text = transcript_text
        db.commit()

        return transcript_text

    except Exception as exc:
        update_job_status(
            db, uuid.UUID(job_id), JobStatus.FAILED,
            error_message=f"Transcription failed: {str(exc)}",
        )
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(
    bind=True,
    name="app.tasks.video_tasks.analyze_clips_task",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def analyze_clips_task(self, job_id: str, transcript_text: str) -> str:
    db = get_sync_db()
    try:
        job_uuid = uuid.UUID(job_id)
        update_job_status(
            db, job_uuid, JobStatus.ANALYZING,
            progress=40.0,
            current_step="GPT-4 analyzing content for clips...",
        )

        import openai

        openai.api_key = settings.OPENAI_API_KEY

        job = db.query(Job).filter(Job.id == job_uuid).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")

        clip_count = job.clip_count
        clip_length = job.clip_length
        min_duration = clip_length - 5
        max_duration = clip_length

        system_prompt = f"""You are an Expert Viral Content Strategist. Your job is to identify exactly {clip_count} video segments that are EXACTLY {clip_length} seconds long (tolerance: {min_duration}-{max_duration} seconds).

CRITICAL REQUIREMENTS:
1. Each clip MUST be {clip_length} seconds long - this is non-negotiable
2. Clips CANNOT overlap - use completely different time ranges
3. Pick segments with clear, complete narratives
4. Start clips at natural transition points when possible

OUTPUT FORMAT - STRICTLY FOLLOW THIS PATTERN:
```
CLIP 1 - [CATEGORY]:
START: [number] END: [number]
```

Example: If clip 1 starts at 0s and should be {clip_length}s long, output START: 0 END: {clip_length}

TRANSCRIPT:
{transcript_text}"""

        user_prompt = f"""Extract exactly {clip_count} clips of {clip_length} seconds each from this transcript.

CRITICAL: Your START and END values must result in clips that are {clip_length} seconds long!
Example: If START is 0, then END must be {clip_length}

Follow the format exactly:
```
CLIP 1 - HOOK FOCUS:
START: [0] END: [{clip_length}]

CLIP 2 - EMOTIONAL PEAK:
START: [X] END: [X+{clip_length}]

CLIP 3 - VIRAL MOMENT:
START: [Y] END: [Y+{clip_length}]
```

TRANSCRIPT:
{transcript_text}"""

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        analysis = response.choices[0].message.content
        job.gpt_analysis = analysis
        db.commit()

        return analysis

    except Exception as exc:
        update_job_status(
            db, uuid.UUID(job_id), JobStatus.FAILED,
            error_message=f"Clip analysis failed: {str(exc)}",
        )
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(
    bind=True,
    name="app.tasks.video_tasks.extract_clips_task",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
)
def extract_clips_task(self, job_id: str, analysis: str) -> list[dict]:
    db = get_sync_db()
    try:
        job_uuid = uuid.UUID(job_id)
        update_job_status(
            db, job_uuid, JobStatus.EXTRACTING_CLIPS,
            progress=55.0,
            current_step="Extracting clip timestamps...",
        )

        import re

        job = db.query(Job).filter(Job.id == job_uuid).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")

        clip_length = job.clip_length
        clip_count = job.clip_count
        min_clip = clip_length - 5
        max_clip = clip_length

        pattern1 = re.compile(r"(?i)start[:\s]*(\d+(?:\.\d+)?)\s*(?:s|sec)?", re.DOTALL)
        pattern2 = re.compile(r"(?i)end[:\s]*(\d+(?:\.\d+)?)\s*(?:s|sec)?", re.DOTALL)

        blocks = re.split(r"(?i)(?:clip|segment)\s*\d+", analysis)

        all_timestamps = []
        clip_num = 1

        for block in blocks:
            if not block.strip():
                continue

            starts = pattern1.findall(block)
            ends = pattern2.findall(block)

            if starts and ends:
                for start, end in zip(starts[:3], ends[:3]):
                    s = float(start)
                    e = float(end)
                    duration = e - s

                    if duration < 3:
                        e = s + clip_length
                        duration = clip_length

                    if min_clip <= duration <= max_clip:
                        all_timestamps.append(
                            {"number": clip_num, "start": s, "end": e, "duration": duration}
                        )
                        clip_num += 1

        if not all_timestamps:
            for block in blocks:
                if not block.strip():
                    continue

                starts = pattern1.findall(block)
                ends = pattern2.findall(block)

                for start, end in zip(starts[:3], ends[:3]):
                    s = float(start)
                    e = float(end)
                    duration = e - s

                    if duration < 3:
                        e = s + clip_length
                        duration = clip_length

                    if duration >= 3:
                        all_timestamps.append(
                            {"number": clip_num, "start": s, "end": e, "duration": duration}
                        )
                        clip_num += 1

        all_timestamps.sort(key=lambda x: x["start"])
        deduplicated = _remove_overlapping(all_timestamps)
        clips_data = deduplicated[:clip_count]

        categories = {1: "Hook Focus", 2: "Emotional Peak", 3: "Viral Moment"}

        for i, clip_info in enumerate(clips_data):
            clip = Clip(
                job_id=job_uuid,
                clip_number=i + 1,
                category=categories.get(i + 1, f"Clip {i + 1}"),
                start_time=clip_info["start"],
                end_time=clip_info["end"],
                duration=clip_info["duration"],
                status=ClipStatus.PENDING,
            )
            db.add(clip)

        db.commit()

        return clips_data

    except Exception as exc:
        update_job_status(
            db, uuid.UUID(job_id), JobStatus.FAILED,
            error_message=f"Clip extraction failed: {str(exc)}",
        )
        raise
    finally:
        db.close()


def _remove_overlapping(clips: list[dict]) -> list[dict]:
    result = []
    for clip in clips:
        is_overlap = False
        for existing in result:
            overlap_start = max(clip["start"], existing["start"])
            overlap_end = min(clip["end"], existing["end"])
            overlap_duration = overlap_end - overlap_start

            if overlap_duration > 0 and (overlap_duration / clip["duration"]) > 0.5:
                is_overlap = True
                break

        if not is_overlap:
            result.append(clip)

    return result


@celery_app.task(
    bind=True,
    name="app.tasks.video_tasks.process_single_clip_task",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
)
def process_single_clip_task(self, job_id: str, clip_id: str) -> dict:
    db = get_sync_db()
    try:
        clip_uuid = uuid.UUID(clip_id)
        job_uuid = uuid.UUID(job_id)

        clip = db.query(Clip).filter(Clip.id == clip_uuid).first()
        if not clip:
            raise ValueError(f"Clip {clip_id} not found")

        job = db.query(Job).filter(Job.id == job_uuid).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")

        update_clip_status(db, clip_uuid, ClipStatus.EXTRACTING)

        from moviepy.editor import VideoFileClip
        import tempfile

        video = VideoFileClip(job.video_path)
        subclip = video.subclip(clip.start_time, clip.end_time)
        clip_path = os.path.join(settings.TEMP_DIR, f"{clip_id}_original.mp4")
        os.makedirs(os.path.dirname(clip_path), exist_ok=True)
        subclip.write_videofile(clip_path, verbose=False, logger=None)
        subclip.close()
        video.close()

        update_clip_status(db, clip_uuid, ClipStatus.EXTRACTING, original_path=clip_path)

        update_clip_status(db, clip_uuid, ClipStatus.CROPPING)

        cropped_path = os.path.join(settings.TEMP_DIR, f"{clip_id}_cropped.mp4")
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from crop_videos import SmartVerticalCropper

            cropper = SmartVerticalCropper(smoothing_factor=0.1)
            success = cropper.crop_video(clip_path, cropped_path)
            if success:
                update_clip_status(db, clip_uuid, ClipStatus.CROPPING, cropped_path=cropped_path)
            else:
                cropped_path = clip_path
        except Exception as e:
            print(f"Cropping failed, using original: {e}")
            cropped_path = clip_path

        update_clip_status(db, clip_uuid, ClipStatus.CAPTIONING)

        final_path = os.path.join(settings.OUTPUT_DIR, f"{clip_id}_final.mp4")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        no_captions = False

        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from captions import create_captions_video

            success = create_captions_video(
                cropped_path,
                final_path,
                api_key=settings.OPENAI_API_KEY,
                style_preset=job.caption_style,
            )
            if not success:
                final_path = cropped_path
                no_captions = True
        except Exception as e:
            print(f"Captioning failed, using cropped: {e}")
            final_path = cropped_path
            no_captions = True

        update_clip_status(
            db,
            clip_uuid,
            ClipStatus.COMPLETED,
            final_path=final_path,
            no_captions=no_captions,
        )

        return {
            "clip_id": clip_id,
            "clip_number": clip.clip_number,
            "category": clip.category,
            "start_time": clip.start_time,
            "end_time": clip.end_time,
            "duration": clip.duration,
            "original_path": clip_path,
            "final_path": final_path,
            "no_captions": no_captions,
        }

    except Exception as exc:
        update_clip_status(
            db, uuid.UUID(clip_id), ClipStatus.FAILED,
            error_message=str(exc),
        )
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(
    bind=True,
    name="app.tasks.video_tasks.vision_analysis_task",
    max_retries=2,
    default_retry_delay=60,
    acks_late=True,
)
def vision_analysis_task(self, job_id: str) -> None:
    db = get_sync_db()
    try:
        job_uuid = uuid.UUID(job_id)
        job = db.query(Job).filter(Job.id == job_uuid).first()
        if not job or not job.use_vision:
            return

        import openai
        import cv2
        import base64

        openai.api_key = settings.OPENAI_API_KEY

        clips = db.query(Clip).filter(Clip.job_id == job_uuid).all()

        for clip in clips:
            if not clip.original_path or not os.path.exists(clip.original_path):
                continue

            frames = _extract_keyframes(clip.original_path, 0, clip.duration, 3)
            if not frames:
                continue

            user_content = []
            positions = ["start", "middle", "end"]
            for i, frame in enumerate(frames):
                user_content.append({"type": "text", "text": f"Frame {i+1} ({positions[i]} of clip)"})
                base64_frame = _encode_frame(frame)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"},
                })

            user_content.append({
                "type": "text",
                "text": f"""Analyze the visual hook potential and respond EXACTLY in this format:
```
VISUAL_HOOK: [1-2 sentence description of the strongest visual hook element]
VIRALITY_SCORE: [0-10 score for visual engagement potential]
ENGAGEMENT_FACTORS: [List of specific visual elements that drive engagement]
```""",
            })

            try:
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an Expert Visual Content Analyst specializing in viral video hooks."},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=300,
                )

                result = _parse_vision_response(response.choices[0].message.content)
                clip.visual_score = result["virality_score"]
                clip.visual_hook = result["visual_hook"]

                text_score = 7.0
                clip.text_score = text_score
                clip.combined_score = (text_score * 0.6) + (result["virality_score"] * 0.4)

                db.commit()

            except Exception as e:
                print(f"Vision analysis failed for clip {clip.id}: {e}")
                continue

        all_clips = db.query(Clip).filter(Clip.job_id == job_uuid).order_by(Clip.combined_score.desc()).all()
        for rank, clip in enumerate(all_clips, 1):
            clip.rank = rank
        db.commit()

    except Exception as exc:
        print(f"Vision analysis task failed: {exc}")
    finally:
        db.close()


def _extract_keyframes(video_path: str, start_time: float, end_time: float, num_frames: int = 3):
    import cv2

    cap = cv2.VideoCapture(video_path)
    duration = end_time - start_time
    frames = []

    for i in range(num_frames):
        timestamp = start_time + (duration * i / (num_frames - 1)) if num_frames > 1 else start_time + duration / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()
    return frames


def _encode_frame(frame) -> str:
    import cv2
    import base64

    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")


def _parse_vision_response(response_text: str) -> dict:
    virality_score = 5
    visual_hook = ""
    engagement_factors = []

    for line in response_text.split("\n"):
        line = line.strip()
        if line.startswith("VISUAL_HOOK:"):
            visual_hook = line.replace("VISUAL_HOOK:", "").strip()
        elif line.startswith("VIRALITY_SCORE:"):
            try:
                score_str = line.replace("VIRALITY_SCORE:", "").strip()
                virality_score = int("".join(filter(str.isdigit, score_str)))
            except Exception:
                pass
        elif line.startswith("ENGAGEMENT_FACTORS:"):
            continue
        elif line.startswith("-") or line.startswith("*"):
            engagement_factors.append(line.lstrip("-* ").strip())

    return {
        "virality_score": virality_score,
        "visual_hook": visual_hook,
        "engagement_factors": engagement_factors,
    }


@celery_app.task(
    bind=True,
    name="app.tasks.video_tasks.finalize_job_task",
    max_retries=1,
    acks_late=True,
)
def finalize_job_task(self, job_id: str, clip_results: list[dict]) -> None:
    db = get_sync_db()
    try:
        job_uuid = uuid.UUID(job_id)
        job = db.query(Job).filter(Job.id == job_uuid).first()
        if not job:
            return

        failed_clips = db.query(Clip).filter(
            Clip.job_id == job_uuid, Clip.status == ClipStatus.FAILED
        ).count()

        if failed_clips > 0 and failed_clips < len(clip_results):
            job.status = JobStatus.COMPLETED
            job.current_step = f"Completed with {failed_clips} failed clip(s)"
        elif failed_clips == len(clip_results):
            job.status = JobStatus.FAILED
            job.error_message = "All clips failed to process"
        else:
            job.status = JobStatus.COMPLETED
            job.current_step = "All clips processed successfully"

        job.progress = 100.0
        job.completed_at = datetime.utcnow()
        db.commit()

    except Exception as exc:
        update_job_status(
            db, uuid.UUID(job_id), JobStatus.FAILED,
            error_message=f"Finalization failed: {str(exc)}",
        )
    finally:
        db.close()


@celery_app.task(
    bind=True,
    name="app.tasks.video_tasks.process_video_pipeline",
    max_retries=1,
    acks_late=True,
)
def process_video_pipeline(self, job_id: str) -> None:
    try:
        pipeline = chain(
            extract_audio_task.s(job_id),
            transcribe_audio_task.s(job_id),
            analyze_clips_task.s(job_id),
            extract_clips_task.s(job_id),
            _process_clips_and_finalize.s(job_id),
        )
        pipeline.apply_async()

    except Exception as exc:
        db = get_sync_db()
        try:
            update_job_status(
                db, uuid.UUID(job_id), JobStatus.FAILED,
                error_message=f"Pipeline failed to start: {str(exc)}",
            )
        finally:
            db.close()


@celery_app.task(
    bind=True,
    name="app.tasks.video_tasks._process_clips_and_finalize",
    max_retries=1,
    acks_late=True,
)
def _process_clips_and_finalize(self, clips_data: list[dict], job_id: str) -> None:
    db = get_sync_db()
    try:
        job_uuid = uuid.UUID(job_id)
        update_job_status(
            db, job_uuid, JobStatus.PROCESSING_CLIPS,
            progress=60.0,
            current_step="Processing clips...",
        )

        clips = db.query(Clip).filter(Clip.job_id == job_uuid).all()

        clip_tasks = []
        for clip in clips:
            clip_tasks.append(process_single_clip_task.s(job_id, str(clip.id)))

        job = db.query(Job).filter(Job.id == job_uuid).first()
        if job and job.use_vision:
            clip_tasks.append(vision_analysis_task.s(job_id))

        callback = finalize_job_task.si(job_id, clips_data)
        chord(group(clip_tasks), callback).apply_async()

    except Exception as exc:
        update_job_status(
            db, uuid.UUID(job_id), JobStatus.FAILED,
            error_message=f"Clip processing failed: {str(exc)}",
        )
    finally:
        db.close()
