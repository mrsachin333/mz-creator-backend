
from __future__ import annotations

import json
import math
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
PROJECT_DIR = DATA_DIR / "projects"

for p in (UPLOAD_DIR, OUTPUT_DIR, PROJECT_DIR):
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="MZ Creator Studio Backend",
    version="2.0.0",
    description="Backend + FFmpeg media engine for MZ Creator Studio",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later: replace with your Netlify domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: dict[str, dict] = {}
JOB_LOCK = threading.Lock()


class ProjectCreate(BaseModel):
    name: str = "Untitled Project"


class ProjectSave(BaseModel):
    name: str
    data: dict


class ProbeJob(BaseModel):
    upload_id: str


class TranscodeJob(BaseModel):
    upload_id: str
    resolution: str = "720p"


class EditJob(BaseModel):
    upload_id: str
    trim_start: float = 0.0
    trim_end: Optional[float] = None

    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    reverse: bool = False

    rotate: int = 0  # supported: 0, 90, 180, 270
    flip_h: bool = False
    flip_v: bool = False

    aspect_ratio: Optional[str] = None  # 9:16, 16:9, 1:1, 4:5
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = Field(default=None, ge=1, le=120)

    brightness: float = Field(default=0.0, ge=-1.0, le=1.0)
    contrast: float = Field(default=1.0, ge=0.1, le=3.0)
    saturation: float = Field(default=1.0, ge=0.0, le=3.0)

    mute: bool = False
    volume: float = Field(default=1.0, ge=0.0, le=3.0)


class MergeJob(BaseModel):
    upload_ids: List[str]
    resolution: str = "720p"


class ExtractAudioJob(BaseModel):
    upload_id: str
    format: str = "mp3"  # mp3 | m4a


class ThumbnailJob(BaseModel):
    upload_id: str
    time_sec: float = 0.0


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def safe_name(name: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    cleaned = "".join(ch if ch in keep else "_" for ch in (name or "file"))
    return cleaned[:120] or "file"


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def update_job(job_id: str, **kwargs):
    with JOB_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)
            JOBS[job_id]["updated_at"] = time.time()


def find_upload(upload_id: str) -> Path:
    matches = list(UPLOAD_DIR.glob(f"{upload_id}__*"))
    if not matches:
        raise FileNotFoundError(upload_id)
    return matches[0]


def probe_file(path: Path) -> dict:
    code, out, err = run_cmd([
        "ffprobe", "-v", "error",
        "-show_entries",
        "format=duration,size,bit_rate:stream=codec_name,codec_type,width,height,r_frame_rate",
        "-of", "json",
        str(path)
    ])
    if code != 0:
        raise RuntimeError(err[-3000:] or "ffprobe failed")
    return json.loads(out)


def aspect_crop_filter(ratio: str) -> str:
    ratio_map = {
        "9:16": "9/16",
        "16:9": "16/9",
        "1:1": "1/1",
        "4:5": "4/5",
    }
    target = ratio_map.get(ratio)
    if not target:
        raise ValueError("Unsupported aspect_ratio")
    # Center crop preserving as much image as possible.
    return (
        f"crop='if(gt(iw/ih,{target}),ih*{target},iw)':"
        f"'if(gt(iw/ih,{target}),ih,iw/{target})'"
    )


def build_video_filters(req: EditJob) -> list[str]:
    vf: list[str] = []

    # Crop first.
    if req.aspect_ratio:
        vf.append(aspect_crop_filter(req.aspect_ratio))

    # Rotate.
    rot = req.rotate % 360
    if rot == 90:
        vf.append("transpose=1")
    elif rot == 180:
        vf += ["hflip", "vflip"]
    elif rot == 270:
        vf.append("transpose=2")
    elif rot != 0:
        raise ValueError("rotate must be 0, 90, 180 or 270")

    if req.flip_h:
        vf.append("hflip")
    if req.flip_v:
        vf.append("vflip")

    # Basic color controls.
    if (
        abs(req.brightness) > 1e-6
        or abs(req.contrast - 1.0) > 1e-6
        or abs(req.saturation - 1.0) > 1e-6
    ):
        vf.append(
            f"eq=brightness={req.brightness}:contrast={req.contrast}:saturation={req.saturation}"
        )

    # Speed.
    if abs(req.speed - 1.0) > 1e-6:
        vf.append(f"setpts=PTS/{req.speed}")

    # Resize.
    if req.width and req.height:
        vf.append(
            f"scale={req.width}:{req.height}:force_original_aspect_ratio=decrease,"
            f"pad={req.width}:{req.height}:(ow-iw)/2:(oh-ih)/2"
        )
    elif req.height:
        vf.append(f"scale=-2:{req.height}:force_original_aspect_ratio=decrease")
    elif req.width:
        vf.append(f"scale={req.width}:-2:force_original_aspect_ratio=decrease")

    if req.fps:
        vf.append(f"fps={req.fps}")

    # H.264 / yuv420p requires even frame dimensions.
    # This is a no-op for normal even-sized video and safely rounds odd dimensions down by 1 px.
    vf.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")

    return vf


def build_audio_filters(req: EditJob) -> list[str]:
    af: list[str] = []

    # Reverse audio when reversing video.
    if req.reverse:
        af.append("areverse")

    # Speed audio. atempo supports 0.5-2.0 per stage, so chain when needed.
    speed = req.speed
    if abs(speed - 1.0) > 1e-6:
        parts = []
        x = speed
        while x > 2.0:
            parts.append("atempo=2.0")
            x /= 2.0
        while x < 0.5:
            parts.append("atempo=0.5")
            x /= 0.5
        parts.append(f"atempo={x:.6f}")
        af += parts

    if abs(req.volume - 1.0) > 1e-6:
        af.append(f"volume={req.volume}")

    return af


def process_probe(job_id: str, req: ProbeJob):
    try:
        update_job(job_id, status="processing", progress=10)
        src = find_upload(req.upload_id)
        result = probe_file(src)
        out_file = OUTPUT_DIR / f"{job_id}.json"
        out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        update_job(
            job_id,
            status="completed",
            progress=100,
            result_file=out_file.name,
            result_type="json",
        )
    except Exception as e:
        update_job(job_id, status="failed", progress=100, error=str(e))


def process_transcode(job_id: str, req: TranscodeJob):
    try:
        update_job(job_id, status="processing", progress=10)
        src = find_upload(req.upload_id)
        height = 1080 if req.resolution == "1080p" else 720
        out_file = OUTPUT_DIR / f"{job_id}.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-vf", f"scale=-2:{height}:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out_file),
        ]
        update_job(job_id, progress=35)
        code, _, err = run_cmd(cmd)
        if code != 0:
            raise RuntimeError(err[-4000:] or "ffmpeg failed")
        update_job(job_id, status="completed", progress=100, result_file=out_file.name, result_type="video")
    except Exception as e:
        update_job(job_id, status="failed", progress=100, error=str(e))


def process_edit(job_id: str, req: EditJob):
    try:
        update_job(job_id, status="processing", progress=5)
        src = find_upload(req.upload_id)
        meta = probe_file(src)
        dur = float(meta.get("format", {}).get("duration") or 0)

        trim_end = req.trim_end if req.trim_end is not None else dur
        trim_start = max(0.0, req.trim_start)
        trim_end = min(trim_end, dur) if dur > 0 else trim_end
        if trim_end <= trim_start:
            raise ValueError("trim_end must be greater than trim_start")

        out_file = OUTPUT_DIR / f"{job_id}.mp4"
        cmd = ["ffmpeg", "-y"]

        # Accurate trim.
        cmd += ["-ss", f"{trim_start:.3f}", "-to", f"{trim_end:.3f}", "-i", str(src)]

        vf = build_video_filters(req)
        af = build_audio_filters(req)

        if req.reverse:
            vf.append("reverse")

        if vf:
            cmd += ["-vf", ",".join(vf)]

        if req.mute:
            cmd += ["-an"]
        elif af:
            cmd += ["-af", ",".join(af)]

        cmd += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "21",
            "-pix_fmt", "yuv420p",
        ]

        if not req.mute:
            cmd += ["-c:a", "aac", "-b:a", "128k"]

        cmd += ["-movflags", "+faststart", str(out_file)]

        update_job(job_id, progress=25)
        code, _, err = run_cmd(cmd)
        if code != 0:
            raise RuntimeError(err[-5000:] or "ffmpeg edit failed")

        update_job(
            job_id,
            status="completed",
            progress=100,
            result_file=out_file.name,
            result_type="video",
        )
    except Exception as e:
        update_job(job_id, status="failed", progress=100, error=str(e))


def process_merge(job_id: str, req: MergeJob):
    try:
        if len(req.upload_ids) < 2:
            raise ValueError("At least 2 uploads are required")

        update_job(job_id, status="processing", progress=5)
        paths = [find_upload(x) for x in req.upload_ids]

        height = 1080 if req.resolution == "1080p" else 720

        # Normalize clips first, then concat.
        normalized: list[Path] = []
        for i, src in enumerate(paths):
            out = OUTPUT_DIR / f"{job_id}_part{i}.mp4"
            cmd = [
                "ffmpeg", "-y", "-i", str(src),
                "-vf", f"scale=-2:{height}:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-r", "30",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "128k",
                "-movflags", "+faststart",
                str(out)
            ]
            code, _, err = run_cmd(cmd)
            if code != 0:
                raise RuntimeError(err[-4000:] or f"Normalize failed for clip {i}")
            normalized.append(out)
            update_job(job_id, progress=10 + int((i + 1) / len(paths) * 55))

        list_file = OUTPUT_DIR / f"{job_id}_concat.txt"
        list_file.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in normalized),
            encoding="utf-8",
        )

        out_file = OUTPUT_DIR / f"{job_id}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-movflags", "+faststart",
            str(out_file)
        ]
        code, _, err = run_cmd(cmd)
        if code != 0:
            raise RuntimeError(err[-4000:] or "Concat failed")

        update_job(job_id, status="completed", progress=100, result_file=out_file.name, result_type="video")
    except Exception as e:
        update_job(job_id, status="failed", progress=100, error=str(e))


def process_extract_audio(job_id: str, req: ExtractAudioJob):
    try:
        src = find_upload(req.upload_id)
        update_job(job_id, status="processing", progress=15)

        fmt = req.format.lower()
        if fmt not in ("mp3", "m4a"):
            raise ValueError("format must be mp3 or m4a")

        out_file = OUTPUT_DIR / f"{job_id}.{fmt}"
        if fmt == "mp3":
            cmd = ["ffmpeg", "-y", "-i", str(src), "-vn", "-c:a", "libmp3lame", "-b:a", "192k", str(out_file)]
        else:
            cmd = ["ffmpeg", "-y", "-i", str(src), "-vn", "-c:a", "aac", "-b:a", "192k", str(out_file)]

        code, _, err = run_cmd(cmd)
        if code != 0:
            raise RuntimeError(err[-3000:] or "Audio extraction failed")

        update_job(job_id, status="completed", progress=100, result_file=out_file.name, result_type="audio")
    except Exception as e:
        update_job(job_id, status="failed", progress=100, error=str(e))


def process_thumbnail(job_id: str, req: ThumbnailJob):
    try:
        src = find_upload(req.upload_id)
        update_job(job_id, status="processing", progress=20)
        out_file = OUTPUT_DIR / f"{job_id}.jpg"

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{max(0.0, req.time_sec):.3f}",
            "-i", str(src),
            "-frames:v", "1",
            "-q:v", "2",
            str(out_file)
        ]
        code, _, err = run_cmd(cmd)
        if code != 0:
            raise RuntimeError(err[-3000:] or "Thumbnail extraction failed")

        update_job(job_id, status="completed", progress=100, result_file=out_file.name, result_type="image")
    except Exception as e:
        update_job(job_id, status="failed", progress=100, error=str(e))


def create_job_record(kind: str, payload: dict) -> str:
    job_id = new_id("job")
    now = time.time()
    JOBS[job_id] = {
        "job_id": job_id,
        "kind": kind,
        "status": "queued",
        "progress": 0,
        "created_at": now,
        "updated_at": now,
        "error": None,
        "result_file": None,
        **payload,
    }
    return job_id


@app.get("/")
def root():
    return {
        "name": "MZ Creator Studio Backend",
        "status": "online",
        "version": "2.0.1",
        "step": "FFmpeg Media Engine",
    }


@app.get("/health")
def health():
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ffprobe_ok = shutil.which("ffprobe") is not None
    return {
        "ok": ffmpeg_ok and ffprobe_ok,
        "ffmpeg": ffmpeg_ok,
        "ffprobe": ffprobe_ok,
        "version": "2.0.1",
    }


@app.post("/uploads")
async def upload_media(file: UploadFile = File(...)):
    upload_id = new_id("upload")
    filename = safe_name(file.filename or "media.bin")
    dst = UPLOAD_DIR / f"{upload_id}__{filename}"

    with dst.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    return {
        "upload_id": upload_id,
        "filename": filename,
        "size": dst.stat().st_size,
    }


@app.get("/uploads/{upload_id}")
def upload_info(upload_id: str):
    try:
        p = find_upload(upload_id)
    except FileNotFoundError:
        raise HTTPException(404, "Upload not found")

    return {
        "upload_id": upload_id,
        "filename": p.name.split("__", 1)[-1],
        "size": p.stat().st_size,
    }


@app.post("/projects")
def create_project(req: ProjectCreate):
    project_id = new_id("project")
    payload = {
        "project_id": project_id,
        "name": req.name,
        "data": {},
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    (PROJECT_DIR / f"{project_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


@app.put("/projects/{project_id}")
def save_project(project_id: str, req: ProjectSave):
    path = PROJECT_DIR / f"{project_id}.json"
    if not path.exists():
        raise HTTPException(404, "Project not found")

    old = json.loads(path.read_text(encoding="utf-8"))
    old["name"] = req.name
    old["data"] = req.data
    old["updated_at"] = time.time()
    path.write_text(json.dumps(old, indent=2), encoding="utf-8")
    return old


@app.get("/projects/{project_id}")
def get_project(project_id: str):
    path = PROJECT_DIR / f"{project_id}.json"
    if not path.exists():
        raise HTTPException(404, "Project not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/jobs/probe")
def create_probe_job(req: ProbeJob, background_tasks: BackgroundTasks):
    try:
        find_upload(req.upload_id)
    except FileNotFoundError:
        raise HTTPException(404, "Upload not found")

    job_id = create_job_record("probe", {"upload_id": req.upload_id})
    background_tasks.add_task(process_probe, job_id, req)
    return JOBS[job_id]


@app.post("/jobs/transcode")
def create_transcode_job(req: TranscodeJob, background_tasks: BackgroundTasks):
    try:
        find_upload(req.upload_id)
    except FileNotFoundError:
        raise HTTPException(404, "Upload not found")

    job_id = create_job_record("transcode", req.model_dump())
    background_tasks.add_task(process_transcode, job_id, req)
    return JOBS[job_id]


@app.post("/jobs/edit")
def create_edit_job(req: EditJob, background_tasks: BackgroundTasks):
    try:
        find_upload(req.upload_id)
    except FileNotFoundError:
        raise HTTPException(404, "Upload not found")

    job_id = create_job_record("edit", req.model_dump())
    background_tasks.add_task(process_edit, job_id, req)
    return JOBS[job_id]


@app.post("/jobs/merge")
def create_merge_job(req: MergeJob, background_tasks: BackgroundTasks):
    for upload_id in req.upload_ids:
        try:
            find_upload(upload_id)
        except FileNotFoundError:
            raise HTTPException(404, f"Upload not found: {upload_id}")

    job_id = create_job_record("merge", req.model_dump())
    background_tasks.add_task(process_merge, job_id, req)
    return JOBS[job_id]


@app.post("/jobs/extract-audio")
def create_extract_audio_job(req: ExtractAudioJob, background_tasks: BackgroundTasks):
    try:
        find_upload(req.upload_id)
    except FileNotFoundError:
        raise HTTPException(404, "Upload not found")

    job_id = create_job_record("extract_audio", req.model_dump())
    background_tasks.add_task(process_extract_audio, job_id, req)
    return JOBS[job_id]


@app.post("/jobs/thumbnail")
def create_thumbnail_job(req: ThumbnailJob, background_tasks: BackgroundTasks):
    try:
        find_upload(req.upload_id)
    except FileNotFoundError:
        raise HTTPException(404, "Upload not found")

    job_id = create_job_record("thumbnail", req.model_dump())
    background_tasks.add_task(process_thumbnail, job_id, req)
    return JOBS[job_id]


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "completed" or not job.get("result_file"):
        raise HTTPException(409, "Result not ready")

    p = OUTPUT_DIR / job["result_file"]
    if not p.exists():
        raise HTTPException(404, "Result file missing")

    media_type = {
        ".json": "application/json",
        ".mp4": "video/mp4",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(p.suffix.lower(), "application/octet-stream")

    return FileResponse(p, media_type=media_type, filename=p.name)
