import os, shutil, uuid
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from count_video import main, get_model

# Settings del DEMO publico (vitrina): rapido y limitado. Editables por variables
# de entorno en HF Spaces (Settings -> Variables) sin tocar codigo.
DEMO_WIDTH = int(os.environ.get("DEMO_WIDTH", "960"))
DEMO_STRIDE = int(os.environ.get("DEMO_STRIDE", "2"))
DEMO_MAX_SECONDS = int(os.environ.get("DEMO_MAX_SECONDS", "30"))

@asynccontextmanager
async def lifespan(app):
    print("Cargando modelo YOLO...")
    get_model()
    print("Modelo listo. Servidor arriba.")
    yield

app = FastAPI(title="CV Counter", lifespan=lifespan)

UPLOAD_DIR, OUTPUT_DIR, EXAMPLES_DIR = "api_uploads", "api_outputs", "examples"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXAMPLES_DIR, exist_ok=True)

# Sirve los videos de ejemplo de la galeria (carpeta examples/)
app.mount("/examples", StaticFiles(directory=EXAMPLES_DIR), name="examples")

@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open("dashboard.html", encoding="utf-8-sig") as f:
        return f.read()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/count")
async def count(
    video: UploadFile = File(...),
    line: int = Form(400),
    width: int = Form(1280),
    clase: Optional[int] = Form(None),
):
    job = uuid.uuid4().hex[:8]
    in_path = os.path.join(UPLOAD_DIR, f"{job}_{video.filename}")
    out_path = os.path.join(OUTPUT_DIR, f"{job}_out.mp4")
    with open(in_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    only = [clase] if clase is not None else None
    counts = main(in_path, out_path,
                  line_start=(0, line), line_end=(width, line),
                  only_classes=only,
                  target_width=DEMO_WIDTH, stride=DEMO_STRIDE, max_seconds=DEMO_MAX_SECONDS)
    return JSONResponse({
        "job_id": job, "in": counts["in"], "out": counts["out"],
        "download": f"/download/{job}",
    })

@app.get("/download/{job}")
def download(job: str):
    out_path = os.path.join(OUTPUT_DIR, f"{job}_out.mp4")
    if not os.path.exists(out_path):
        return JSONResponse({"error": "no encontrado"}, status_code=404)
    return FileResponse(out_path, media_type="video/mp4", filename=f"{job}_out.mp4")