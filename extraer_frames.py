import cv2, os
os.makedirs('docs', exist_ok=True)
for src, dst in [('examples/retail.mp4','docs/retail.jpg'),
                 ('examples/traffic.mp4','docs/traffic.jpg'),
                 ('examples/apples.mp4','docs/apples.jpg')]:
    v = cv2.VideoCapture(src)
    n = int(v.get(7)); v.set(1, n//3)
    ok, f = v.read()
    if ok: cv2.imwrite(dst, f); print('OK', dst)
