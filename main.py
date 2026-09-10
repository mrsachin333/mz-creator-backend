
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
PROJECT_DIR = DATA_DIR / "projects"

for p in (UPLOAD_DIR, OUTPUT_DIR, PROJECT_DIR):
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="MZ Creator Studio Backend",
    version="1.0.0",
    description="Backend foundation for MZ Creator Studio",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock this down to your Netlify domain later
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


class JobCreate(BaseModel):
    upload_id: str
    action: str = "probe"   # probe | transcode
    resolution: str = "720p"  # used for transcode


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def safe_name(name: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    cleaned = "".join(ch if ch in keep else "_" for ch in (name or "file"))
    return cleaned[:120] or "file"


def job_update(job_id: str, **kwargs):
    with JOB_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)
            JOBS[job_id]["updated_at"] = time.time()


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def find_upload(upload_id: str) -> Path:
    matches = list(UPLOAD_DIR.glob(f"{upload_id}__*"))
    if not matches:
        raise FileNotFoundError(upload_id)
    return matches[0]


def process_job(job_id: str, req: JobCreate):
    try:
        src = find_upload(req.upload_id)
        job_update(job_id, status="processing", progress=5)

        if req.action == "probe":
            code, out, err = run_cmd([
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration,size,bit_rate:stream=codec_name,width,height,r_frame_rate",
                "-of", "json",
                str(src),
            ])
            if code != 0:
                raise RuntimeError(err[-2000:] or "ffprobe failed")

            result = json.loads(out)
            out_file = OUTPUT_DIR / f"{job_id}.json"
            out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

            job_update(
                job_id,
                status="completed",
                progress=100,
                result_type="json",
                result_file=out_file.name,
            )
            return

        if req.action == "transcode":
            height = 1080 if req.resolution == "1080p" else 720
            out_file = OUTPUT_DIR / f"{job_id}.mp4"

            # Keep aspect ratio, ensure even width, standard H.264/AAC MP4.
            vf = f"scale=-2:{height}:force_original_aspect_ratio=decrease"
            cmd = [
                "ffmpeg", "-y",
                "-i", str(src),
                "-vf", vf,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "22",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                str(out_file),
            ]
            job_update(job_id, progress=20)
            code, out, err = run_cmd(cmd)
            if code != 0:
                raise RuntimeError(err[-3000:] or "ffmpeg failed")

            job_update(
                job_id,
                status="completed",
                progress=100,
                result_type="video",
                result_file=out_file.name,
            )
            return

        raise RuntimeError(f"Unknown action: {req.action}")

    except Exception as e:
        job_update(job_id, status="failed", progress=100, error=str(e))


@app.get("/")
def root():
    return {
        "name": "MZ Creator Studio Backend",
        "status": "online",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ffprobe_ok = shutil.which("ffprobe") is not None
    return {
        "ok": ffmpeg_ok and ffprobe_ok,
        "ffmpeg": ffmpeg_ok,
        "ffprobe": ffprobe_ok,
        "uploads_dir": str(UPLOAD_DIR),
        "outputs_dir": str(OUTPUT_DIR),
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
    (PROJECT_DIR / f"{project_id}.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
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


@app.post("/jobs")
def create_job(req: JobCreate, background_tasks: BackgroundTasks):
    try:
        find_upload(req.upload_id)
    except FileNotFoundError:
        raise HTTPException(404, "Upload not found")

    job_id = new_id("job")
    now = time.time()
    JOBS[job_id] = {
        "job_id": job_id,
        "upload_id": req.upload_id,
        "action": req.action,
        "resolution": req.resolution,
        "status": "queued",
        "progress": 0,
        "created_at": now,
        "updated_at": now,
        "error": None,
        "result_file": None,
    }
    background_tasks.add_task(process_job, job_id, req)
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

    media_type = "application/json" if p.suffix == ".json" else "video/mp4"
    return FileResponse(p, media_type=media_type, filename=p.name)
