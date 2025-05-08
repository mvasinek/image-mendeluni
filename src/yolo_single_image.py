import cv2
from ultralytics import YOLO

# Načti model YOLO
yolo = YOLO('yolov8s.pt')

# Načti obrázek (změň cestu podle potřeby)
image_path = 'obrazek.jpg'
frame = cv2.imread(image_path)

if frame is None:
    raise FileNotFoundError(f'Obrázek "{image_path}" nebyl nalezen.')

# Funkce pro výběr barvy podle třídy
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] *
             (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

# Spusť detekci
results = yolo.predict(frame)

# Zpracuj výsledky
for result in results:
    classes_names = result.names

    for box in result.boxes:
        if box.conf[0] > 0.4:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            class_name = classes_names[cls]
            colour = getColours(cls)

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

# Zobraz výsledek
cv2.imshow('Detekce', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()