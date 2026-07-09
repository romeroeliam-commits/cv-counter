import torch
torch.set_num_threads(8)
from ultralytics import YOLO

model = YOLO(r'runs\detect\runs_manzanas\modelo_final\weights\best.pt')
results = model.predict(
    source='dataset_manzanas/test/images',
    conf=0.35,
    save=True,
    project='runs_manzanas',
    name='prediccion_test',
    exist_ok=True,
)
print('Listo. Imagenes anotadas en: runs\\detect\\runs_manzanas\\prediccion_test')
