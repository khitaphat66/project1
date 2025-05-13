from picamera2 import Picamera2
from libcamera import Transform
import cv2
import time

# โหลดโมเดล Haar cascade สำหรับใบหน้าและดวงตา
eye_cascade = cv2.CascadeClassifier('/home/user/haarcascades/haarcascade_eye.xml')
face_ascade = cv2.CascadeClassifier("/home/user/haarcascades/haarcascade_frontalface_default.xml")

picam2 = Picamera2()
#config = picam2.create_preview_configuration(main={"size": (640, 480)})
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    transform=Transform(hflip=0, vflip=0)
    )
picam2.configure(config)
picam2.start()

time.sleep(2)
picam2.set_controls({"AwbEnable": True})

def is_valid_eye(w,h):
    aspect_ratio = w/h
    return 0.5 < aspect_ratio < 2.5 and 10 < w < 150 and 10 < h < 150

while True:
    frame = picam2.capture_array("main")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # แปลงภาพเป็น grayscale
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)  # ตรวจจับดวงตา
    
    for (x, y, w, h) in eyes:
        if is_valid_eye(w,h):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # วาดกรอบรอบตา
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)  # วาดกรอบรอบตา

    cv2.imshow('Eye Detection', frame)  # แสดงผลภาพ

    if cv2.waitKey(1) & 0xFF == ord('q'):  # กด 'q' เพื่อออก
        break

cv2.destroyAllWindows()



---------------------------------------------------------
from picamera2 import Picamera2
from libcamera import Transform
import cv2
import time

# โหลด Haar cascades
eye_cascade = cv2.CascadeClassifier('/home/user/haarcascades/haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier("/home/user/haarcascades/haarcascade_frontalface_default.xml")

# ตั้งค่ากล้อง
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    transform=Transform(hflip=0, vflip=0)
)
picam2.configure(config)
picam2.start()

time.sleep(2)
picam2.set_controls({"AwbEnable": True})

# ฟังก์ชันกรองดวงตา
def is_valid_eye(w, h):
    aspect_ratio = w / h
    return 0.5 < aspect_ratio < 2.5 and 10 < w < 150 and 10 < h < 150

while True:
    frame = picam2.capture_array("main")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้า
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # วาดกรอบใบหน้า
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # ตรวจจับดวงตาในใบหน้า
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            if is_valid_eye(ew, eh):
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            else:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 1)

    # แสดงภาพ
    cv2.imshow('Face & Eye Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
--------------------------------------------
from picamera2 import Picamera2
import cv2
import numpy as np
from time import sleep
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gpiozero import LED, Button

# ปุ่ม & ไฟ
button = Button(18)
exit_button = Button(25)
red = LED(17)
green = LED(27)
yellow = LED(22)
blue = LED(5)

# ไฟกระพริบเริ่มต้น
for _ in range(4):
    red.on(); green.on(); yellow.on(); blue.on()
    sleep(0.5)
    red.off(); green.off(); yellow.off(); blue.off()
    sleep(0.5)

# โหลดโมเดลทำนายโรค
loaded_model = load_model('/boot/overlays/cataract_model.h5')

# โหลด Haar cascade สำหรับดวงตา
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# กล้อง
camera = Picamera2()
camera.preview_configuration.main.size = (640, 480)
camera.configure("preview")
camera.start()
print("กดสวิตช์สีขาวเพื่อถ่าย หรือกดสวิตช์สีเหลืองเพื่อออก")
blue.on()

while True:
    frame = camera.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w, _ = frame.shape
    x1 = w//2 - 200
    y1 = h//2 - 200
    x2 = w//2 + 200
    y2 = h//2 + 200
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Camera Preview", frame)

    # ถ่ายภาพเมื่อกดปุ่ม
    if button.is_pressed:
        blue.off()
        yellow.blink(on_time=0.5, off_time=0.5)

        capture = frame.copy()
        img_crop = capture[y1:y2, x1:x2]

        gray_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray_crop, scaleFactor=1.3, minNeighbors=5)

        if len(eyes) == 0:
            print("ไม่พบดวงตาในภาพ")
            yellow.off()
            red.blink(on_time=0.2, off_time=0.2)
            sleep(5)
            red.off()
            blue.on()
            continue

        print("พบดวงตาในภาพ กำลังวิเคราะห์...")

        img = cv2.resize(img_crop, (150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = loaded_model.predict(img_array)
        print("Prediction values:", prediction)
        yellow.off()

        if np.argmax(prediction) == 0:
            result_text = "ผลลัพธ์: เป็น Cataract"
            print(result_text)
            red.on()
        else:
            result_text = "ผลลัพธ์: เป็น Normal"
            print(result_text)
            green.on()

        sleep(5)
        red.off()
        green.off()
        blue.on()

    # ออกจากโปรแกรม
    elif exit_button.is_pressed:
        print("ออกจากโปรแกรม")
        break

# ปิดการทำงาน
blue.off()
red.off()
green.off()
camera.stop()
cv2.destroyAllWindows()
