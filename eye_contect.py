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
face_cascade = cv2.CascadeClassifier('/home/user/haarcascades/haarcascade_frontalface_default.xml')

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
    return 0.3 < aspect_ratio < 3.5 and 5 < w < 200 and 5 < h < 200

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
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        for (ex, ey, ew, eh) in eyes:
            if ey + eh < h // 2 and is_valid_eye(ew, eh):
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
from libcamera import Transform
import cv2
import numpy as np
from time import sleep
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gpiozero import LED, Button

# ปุ่มและ LED
button = Button(18)
exit_button = Button(26)
red = LED(17)
green = LED(27)
yellow = LED(23)
blue = LED(24)

# กระพริบไฟ LED ตอนเริ่มต้น (เหมือนเดิม)
for _ in range(4):
    red.on()
    green.on()
    yellow.on()
    blue.on()
    sleep(0.5)
    red.off()
    green.off()
    yellow.off()
    blue.off()
    sleep(0.5)

# โหลดโมเดล cataract
loaded_model = load_model('/boot/overlays/cataract_model.h5')

# โหลด Haar cascades สำหรับตรวจจับใบหน้าและดวงตา
face_cascade = cv2.CascadeClassifier('/home/user/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/user/haarcascades/haarcascade_eye.xml')

# ฟังก์ชันตรวจสอบว่าภาพมีใบหน้าและดวงตาที่สมเหตุสมผลหรือไม่
def contains_face_and_eyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            aspect_ratio = ew / float(eh)
            if 0.3 < aspect_ratio < 3.5 and 5 < ew < 200 and 5 < eh < 200:
                return True
    return False

# ตั้งค่ากล้อง
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    transform=Transform(hflip=0, vflip=0)
)
picam2.configure(config)
picam2.start()
picam2.set_controls({"AwbEnable": True})

print("กดปุ่มสีขาวเพื่อถ่ายภาพ / กดปุ่มสีเหลืองเพื่อออก")

blue.on()

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # วาดกรอบโซนที่ต้องการจับภาพ
    h, w, _ = frame.shape
    x1, y1 = w // 2 - 200, h // 2 - 200
    x2, y2 = w // 2 + 200, h // 2 + 200
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Live Preview", frame)

    key = cv2.waitKey(1) & 0xFF

    # ถ่ายภาพเมื่อกดปุ่ม
    if button.is_pressed:
        blue.off()
        yellow.blink(on_time=0.3, off_time=0.3)

        capture = frame.copy()
        img_crop = capture[y1:y2, x1:x2]

        # แปลงภาพให้เข้ากับโมเดล
        img = cv2.resize(img_crop, (150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        if contains_face_and_eyes(img_crop):
            prediction = loaded_model.predict(img_array)
            print("Prediction values:", prediction)
            if np.argmax(prediction) == 0:
                result_text = "ผลลัพธ์: เป็น Cataract"
                red.on()
            else:
                result_text = "ผลลัพธ์: เป็น Normal"
                green.on()
        else:
            result_text = "ไม่พบตาที่สมเหตุสมผล → แสดงว่าเป็น Normal"
            green.on()

        print(result_text)
        sleep(5)

        # ปิดไฟ LED และเปิดกล้องใหม่
        red.off()
        green.off()
        yellow.off()
        blue.on()

    elif exit_button.is_pressed:
        print("ออกจากโปรแกรม")
        break

# ปิดระบบทั้งหมด
for led in [red, green, yellow, blue]:
    led.off()
picam2.stop()
cv2.destroyAllWindows()

----------------------------------
from picamera2 import Picamera2
from libcamera import Transform
import cv2
import numpy as np
from time import sleep
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gpiozero import LED, Button

# โหลด Haar cascades
eye_cascade = cv2.CascadeClassifier('/home/user/haarcascades/haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('/home/user/haarcascades/haarcascade_frontalface_default.xml')

# ปุ่มและ LED
button = Button(18)
exit_but = Button(26)
red = LED(17)
green = LED(27)
yellow = LED(23)
blue = LED(24)

# โหลดโมเดล
loaded_model = load_model('/boot/overlays/cataract_model.h5')

# แสดงไฟสถานะเริ่มต้น
for _ in range(4):
    red.on()
    green.on()
    yellow.on()
    blue.on()
    sleep(0.5)
    red.off()
    green.off()
    yellow.off()
    blue.off()
    sleep(0.5)

# ตั้งค่ากล้อง
camera = Picamera2()
config = camera.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    transform=Transform(hflip=0, vflip=0)
)
camera.configure(config)
camera.start()
sleep(2)
camera.set_controls({"AwbEnable": True})

print("กดปุ่มสีขาวเพื่อถ่าย หรือปุ่มสีเหลืองเพื่อออก")
blue.on()

# ฟังก์ชันกรองดวงตา
def is_valid_eye(w, h):
    aspect_ratio = w / h
    return 0.3 < aspect_ratio < 3.5 and 5 < w < 200 and 5 < h < 200

while True:
    frame = camera.capture_array("main")
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้า
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            if ey + eh < h // 2 and is_valid_eye(ew, eh):
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            else:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 1)

    # วาดกรอบกลางภาพสำหรับวางตา
    h, w, _ = frame.shape
    x1 = w//2 - 200
    y1 = h//2 - 200
    x2 = w//2 + 200
    y2 = h//2 + 200
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.imshow("Face & Eye Detection + Cataract Classification", frame)
    cv2.waitKey(1)

    if button.is_pressed:
        blue.off()
        yellow.blink(on_time=0.5, off_time=0.5)

        capture = frame.copy()
        img_crop = capture[y1:y2, x1:x2]

        if img_crop.shape[2] == 4:
            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGRA2BGR)

        img = cv2.resize(img_crop, (150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = loaded_model.predict(img_array)
        print("Prediction values:", prediction)
        yellow.off()

        if np.argmax(prediction) == 0:
            result_text = "ผลลัพธ์: เป็น Cataract"
            red.on()
        else:
            result_text = "ผลลัพธ์: เป็น Normal"
            green.on()

        print(result_text)
        sleep(5)
        red.off()
        green.off()
        blue.on()

    elif exit_but.is_pressed:
        print("exit")
        break

# ปิดการทำงาน
blue.off()
red.off()
green.off()
camera.stop()
cv2.destroyAllWindows()
