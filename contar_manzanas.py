"""Deteccion y conteo de manzanas en video con modelo custom (best.pt).
En vez de linea In/Out, muestra: manzanas visibles ahora + maximo detectado.
Salida en H.264 (reproducible en navegador). Uso:
  python contar_manzanas.py apple.mp4 examples/apples.mp4
"""
import os, argparse, subprocess
import torch
torch.set_num_threads(8)
import cv2
import supervision as sv
from ultralytics import YOLO
import imageio_ffmpeg

def _to_h264(src, dst):
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        subprocess.run([exe, "-y", "-i", src, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst],
                       check=True, capture_output=True)
        os.remove(src)
    except Exception as e:
        if os.path.exists(src):
            os.replace(src, dst)
        print("Aviso: transcode H.264 fallo, se usa el original:", e)

def main(source_path, target_path, weights=r"runs\detect\runs_manzanas\modelo_final\weights\best.pt",
         conf=0.35, target_width=960, stride=1, max_seconds=None):
    model = YOLO(weights)

    info = sv.VideoInfo.from_video_path(source_path)
    scale = target_width / info.width if info.width > target_width else 1.0
    out_w, out_h = int(info.width * scale), int(info.height * scale)
    out_fps = max(1, round(info.fps / stride))
    out_info = sv.VideoInfo(width=out_w, height=out_h, fps=out_fps, total_frames=info.total_frames)
    end = int(max_seconds * info.fps) if max_seconds else None

    box_annotator = sv.BoxAnnotator(thickness=2)

    max_count = 0
    tmp_path = target_path + ".tmp.mp4"
    with sv.VideoSink(tmp_path, out_info) as sink:
        for frame in sv.get_video_frames_generator(source_path, stride=stride, end=end):
            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h))
            results = model(frame, conf=conf, imgsz=960, augment=True, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            n = len(detections)
            max_count = max(max_count, n)

            out = box_annotator.annotate(frame.copy(), detections=detections)

            # panel de conteo arriba a la izquierda
            label1 = f"Manzanas visibles: {n}"
            label2 = f"Max detectadas: {max_count}"
            (w1, h1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            (w2, h2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            pw = max(w1, w2) + 24
            cv2.rectangle(out, (10, 10), (10 + pw, 10 + h1 + h2 + 30), (20, 15, 30), -1)
            cv2.putText(out, label1, (22, 10 + h1 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 190, 120), 2, cv2.LINE_AA)
            cv2.putText(out, label2, (22, 10 + h1 + h2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 139, 167), 2, cv2.LINE_AA)

            sink.write_frame(out)

    _to_h264(tmp_path, target_path)
    print(f"OK -> {target_path} | Max manzanas detectadas en un frame: {max_count} | {out_w}x{out_h} H264")
    return max_count

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("target")
    p.add_argument("--weights", default=r"runs\detect\runs_manzanas\modelo_final\weights\best.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--target-width", type=int, default=960)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-seconds", type=int, default=None)
    args = p.parse_args()
    main(args.source, args.target, weights=args.weights, conf=args.conf,
         target_width=args.target_width, stride=args.stride, max_seconds=args.max_seconds)
