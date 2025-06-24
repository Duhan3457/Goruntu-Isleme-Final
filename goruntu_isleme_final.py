import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Mozaik uygulama fonksiyonu
def apply_mosaic(image, face_landmarks):
    h, w, _ = image.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]

    # Noktalardan kontur oluştur (dış hatlar)
    hull = cv2.convexHull(np.array(points))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Yüz bölgesini al
    face_region = cv2.bitwise_and(image, image, mask=mask)

    # Mozaik için küçült-büyüt işlemi
    small = cv2.resize(face_region, (0, 0), fx=0.06, fy=0.06)
    mosaic = cv2.resize(small, (face_region.shape[1], face_region.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Mozaik sadece yüz bölgesine uygulanacak
    background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    result = cv2.add(background, cv2.bitwise_and(mosaic, mosaic, mask=mask))
    return result

# Modeli yükle
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Kamerayı başlat
cam = cv2.VideoCapture(0)
print("Kameradan yüz aranıyor. Çıkmak için 'q' tuşuna bas.")

while cam.isOpened():
    basari, frame = cam.read()
    if not basari:
        break

    # BGR'den RGB'ye çevir
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Yüz tespiti yap
    detection_result = detector.detect(mp_image)

    # Yüz varsa mozaik uygula
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            frame_rgb = apply_mosaic(frame_rgb, face_landmarks)

    # Göster
    cv2.imshow("Yüz Mozaik", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
