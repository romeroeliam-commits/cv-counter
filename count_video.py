"""Conteo de objetos cruzando una linea. Optimizado: modelo 1x + resolucion reducida.
Salida en H.264 (reproducible en navegadores). Por defecto FULL quality (trabajo real);
para el demo publico la API pasa settings mas rapidos/limitados."""
import os, json, csv, argparse, subprocess
import cv2
import supervision as sv
from ultralytics import YOLO
import imageio_ffmpeg

_MODEL = None
def get_model(weights="yolov8n.pt"):
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO(weights)
    return _MODEL

def _to_h264(src, dst):
    """Transcodifica a H.264 + yuv420p para que cualquier navegador lo reproduzca."""
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        subprocess.run([exe, "-y", "-i", src, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst],
                       check=True, capture_output=True)
        os.remove(src)
    except Exception as e:
        # si falla el transcode, al menos dejamos el archivo original
        if os.path.exists(src):
            os.replace(src, dst)
        print("Aviso: transcode H.264 fallo, se usa el original:", e)

def main(source_path, target_path,
         line_start=(0, 300), line_end=(1280, 300),
         only_classes=None, target_width=1280, stride=1, max_seconds=None):
    model = get_model()

    info = sv.VideoInfo.from_video_path(source_path)
    scale = target_width / info.width if info.width > target_width else 1.0
    out_w, out_h = int(info.width * scale), int(info.height * scale)
    out_fps = max(1, round(info.fps / stride))
    out_info = sv.VideoInfo(width=out_w, height=out_h, fps=out_fps, total_frames=info.total_frames)

    end = int(max_seconds * info.fps) if max_seconds else None

    ls = (int(line_start[0] * scale), int(line_start[1] * scale))
    le = (int(line_end[0] * scale), int(line_end[1] * scale))
    line_zone = sv.LineZone(start=sv.Point(*ls), end=sv.Point(*le))

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    line_annotator = sv.LineZoneAnnotator()

    tmp_path = target_path + ".tmp.mp4"
    with sv.VideoSink(tmp_path, out_info) as sink:
        for frame in sv.get_video_frames_generator(source_path, stride=stride, end=end):
            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h))
            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            if only_classes is not None:
                detections = detections[[c in only_classes for c in detections.class_id]]
            detections = tracker.update_with_detections(detections)
            line_zone.trigger(detections)
            if detections.tracker_id is None or len(detections) == 0:
                labels = []
            else:
                labels = [f"#{tid} {name}"
                          for tid, name in zip(detections.tracker_id, detections["class_name"])]
            out = box_annotator.annotate(frame.copy(), detections=detections)
            out = label_annotator.annotate(out, detections=detections, labels=labels)
            out = line_annotator.annotate(out, line_counter=line_zone)
            sink.write_frame(out)

    _to_h264(tmp_path, target_path)  # <- salida final en H.264

    counts = {"in": int(line_zone.in_count), "out": int(line_zone.out_count)}
    with open("counts.json", "w") as f:
        json.dump(counts, f, indent=2)
    with open("counts.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["direction", "count"])
        w.writerow(["in", counts["in"]]); w.writerow(["out", counts["out"]])
    print(f"OK -> {target_path} | In: {counts['in']} Out: {counts['out']} | {out_w}x{out_h} stride={stride} H264")
    return counts

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source"); p.add_argument("target")
    p.add_argument("--line", type=int, default=400)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--clase", type=int, default=None)
    p.add_argument("--target-width", type=int, default=1280)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-seconds", type=int, default=None)
    args = p.parse_args()
    only = [args.clase] if args.clase is not None else None
    main(args.source, args.target,
         line_start=(0, args.line), line_end=(args.width, args.line),
         only_classes=only, target_width=args.target_width,
         stride=args.stride, max_seconds=args.max_seconds)