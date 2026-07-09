import torch
torch.set_num_threads(8)
from ultralytics import YOLO
import time

model = YOLO('yolov8n.pt')

start = time.time()
model.train(
    data='dataset_manzanas/data.yaml',
    epochs=40,
    imgsz=640,
    batch=4,
    device='cpu',
    workers=4,
    patience=15,          # corta si no mejora en 15 epocas (ahorra tiempo)
    project='runs_manzanas',
    name='modelo_final',
    exist_ok=True,
)
print(f'--- ENTRENAMIENTO COMPLETO: {(time.time()-start)/60:.1f} minutos ---')
print('Modelo guardado en: runs_manzanas/modelo_final/weights/best.pt')
