# CV Counter — Video Analytics & Object Counting as a Service

Detect, track and count objects in video with AI — people, vehicles, or **any custom object trained on demand**.

**🔴 Live demo:** [xeliamx-cv-counter.hf.space](https://xeliamx-cv-counter.hf.space) — watch pre-processed examples or upload your own clip.

---

## What it does

Upload a video → the service detects and tracks every object of the selected class, counts line crossings (In/Out) or visible instances, and returns an **annotated video + counts (CSV/JSON)**.

| Vertical | Scenario | Result |
|---|---|---|
| **Retail** | People flow at a store entrance (security cam) | In: 5 / Out: 3 |
| **Traffic** | Vehicle counting on a highway | In: 14 / Out: 0 |
| **Agro** | Harvest estimation with a **custom-trained apple detector** | 24 apples detected (max/frame) |

The agro demo runs on a model I trained from scratch (YOLOv8, 40 epochs, **mAP50 0.94**) — proof that the pipeline works for *any* object a client needs, not just the standard classes.

## Stack

- **Detection:** Ultralytics YOLOv8 (pre-trained + custom-trained models)
- **Tracking:** ByteTrack via [Supervision](https://github.com/roboflow/supervision)
- **Counting:** line-crossing zones (In/Out) or per-frame instance counting
- **API & UI:** FastAPI + vanilla-JS dashboard (drag & drop, auto resolution detection)
- **Video:** OpenCV, H.264 output via imageio-ffmpeg (browser-playable)
- **Deploy:** Docker on Hugging Face Spaces

## Run locally

```bash
git clone https://github.com/xeliamx/cv-counter.git
cd cv-counter
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn api:app --reload
# open http://localhost:8000
```

CLI usage (no UI):

```bash
# count people crossing a line
python count_video.py input.mp4 output.mp4 --clase 0 --line 540 --width 1920

# count apples with the custom model
python contar_manzanas.py orchard.mp4 output.mp4 --conf 0.25
```

## Training a custom detector

`train_manzanas.py` shows the full recipe used for the apple model: dataset in YOLO format (Roboflow), CPU-friendly settings (`torch.set_num_threads`, small batch, early stopping via `patience`). Swap the dataset and you get a detector for **your** object — products, livestock, defects, PPE, anything.

## Services

I build this as a freelance service:

- **Count anything in your footage** — annotated video + Excel + 1-page report, 48–72h turnaround.
- **Custom object detector** — trained on your object, delivered as a working system.

📩 Try the [live demo](https://xeliamx-cv-counter.hf.space) with your own clip, or reach out for a free test on a short sample.

## License

MIT
