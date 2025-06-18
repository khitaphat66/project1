from picamera2 import Picamera2
from libcamera import Transform
import cv2
import numpy as np
from time import sleep
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gpiozero import LED, Button

button = Button(18)
exit_but = Button(26)
red = LED(17)
green = LED(27)
yellow = LED(23)
blue = LED(25)

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
    
# โหลดโมเดล Cataract
loaded_model = load_model('/boot/overlays/cataract_model.h5')

# โหลด Haar Cascade สำหรับตา
eye_cascade = cv2.CascadeClassifier('/home/user/haarcascades/haarcascade_eye.xml')

# ตั้งค่ากล้อง
camera = Picamera2()
camera.preview_configuration.main.size = (640, 480)
camera.configure("preview")
camera.start()
print("กดปุ่มสีขาวเพื่อถ่าย หรือปุ่มสีเหลืองเพื่อออก")
blue.on()

while True:
    frame = camera.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # แปลงเป็นภาพ grayscale เพื่อหา eye
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ตรวจจับดวงตา
    eyes = eye_cascade.detectMultiScale(gray, 1.05, 10, minSize=(30,30))
    eye_detected = False
    
    # วาดกรอบรอบดวงตาที่เจอ
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        eye_detected = True
    
    cv2.imshow("Camera Preview", frame)
    key = cv2.waitKey(1)
    
    if button.is_pressed:
        blue.off()
        yellow.blink(on_time=0.5, off_time=0.5)
        
        if eye_detected:
            cataract_found = False
            # ใช้เฉพาะดวงตาแรกที่เจอ
            (ex, ey, ew, eh) = eyes[0]
            eye_crop = frame[ey:ey+eh, ex:ex+ew]
        
            # resize สำหรับโมเดล
            img = cv2.resize(eye_crop, (150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
        
            # ทำนาย
            prediction = loaded_model.predict(img_array)
            print("Prediction values:", prediction)
        
                if (prediction[0, 0] >= 0.1):
                    cataract_found = True

            yellow.off()

            if cataract_found:
                result_text = "ผลลัพธ์: Cataract"
                red.on()
            else:
                result_text = "ผลลัพธ์: Normalิ"
                green.on()
        else:
            result_text = "ไม่พบตาชัดเจน"
            yellow.off()
            green.blink(on_time=0.5, off_time=0.5)

        print(result_text)
        sleep(5)
        red.off()
        green.off()
        blue.on()

    elif exit_but.is_pressed:
        print("exit")
        break

blue.off()
red.off()
green.off()
camera.stop()
cv2.destroyAllWindows()
-------------------------------------------------------
from picamera2 import Picamera2
from libcamera import Transform
import cv2
import numpy as np
from time import sleep
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gpiozero import LED, Button

button = Button(18)
exit_but = Button(26)
red = LED(17)
green = LED(27)
yellow = LED(23)
blue = LED(25)

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
    
# โหลดโมเดล Cataract
loaded_model = load_model('/boot/overlays/cataract_model.h5')

# โหลด Haar Cascade สำหรับตา
eye_cascade = cv2.CascadeClassifier('/home/user/haarcascades/haarcascade_eye.xml')

# ตั้งค่ากล้อง
camera = Picamera2()
camera.preview_configuration.main.size = (640, 480)
camera.configure("preview")
camera.start()
print("กดปุ่มสีขาวเพื่อถ่าย หรือปุ่มสีเหลืองเพื่อออก")
blue.on()

while True:
    frame = camera.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # แปลงเป็นภาพ grayscale เพื่อหา eye
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ตรวจจับดวงตา
    eyes = eye_cascade.detectMultiScale(gray, 1.05, 10, minSize=(30,30))
    eye_detected = False
    
    # วาดกรอบรอบดวงตาที่เจอ
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        eye_detected = True
    
    cv2.imshow("Camera Preview", frame)
    key = cv2.waitKey(1)
    
    if button.is_pressed:
        blue.off()
        yellow.blink(on_time=0.5, off_time=0.5)
        
        if eye_detected:
            (ex, ey, ew, eh) = eyes[0]
            eye_crop = frame[ey:ey+eh, ex:ex+ew]
        
            # resize สำหรับโมเดล
            img = cv2.resize(eye_crop, (150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
        
            # ทำนาย
            prediction = loaded_model.predict(img_array)
            print("Prediction values:", prediction)
        
    
            yellow.off()

            if (prediction[0, 0] >= 0.1):
                result_text = "ผลลัพธ์: Cataract"
                red.on()
            else:
                result_text = "ผลลัพธ์: Normalิ"
                green.on()
        else:
            result_text = "ไม่พบตาชัดเจน"
            yellow.off()
            green.blink(on_time=0.5, off_time=0.5)

        print(result_text)
        sleep(5)
        red.off()
        green.off()
        blue.on()

    elif exit_but.is_pressed:
        print("exit")
        break

blue.off()
red.off()
green.off()
camera.stop()
cv2.destroyAllWindows()
---------------------
# -*- coding: utf-8 -*-
"""
กล้อง Raspberry Pi + Mediapipe FaceMesh → ครอปดวงตา → โมเดล CNN จำแนกต้อกระจก
ทดแทน Haar Cascade เดิม ด้วยความแม่นยำสูงขึ้น
"""

from picamera2 import Picamera2
import cv2
import numpy as np
from time import sleep
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gpiozero import LED, Button
import mediapipe as mp

# ----------------- GPIO -----------------
button    = Button(18)   # ปุ่มขาว  กดถ่าย
exit_but  = Button(26)   # ปุ่มเหลือง ออก
red   = LED(17)          # ผลลัพธ์ Cataract
green = LED(27)          # ผลลัพธ์ Normal
yellow= LED(23)          # กำลังประมวลผล / แจ้งไม่พบตา
blue  = LED(25)          # สถานะพร้อมถ่าย

# เอฟเฟ็กต์ LED ตอนบูต
for _ in range(4):
    red.on(); green.on(); yellow.on(); blue.on();  sleep(0.5)
    red.off(); green.off(); yellow.off(); blue.off(); sleep(0.5)

# ----------------- โมเดล -----------------
model = load_model('/boot/overlays/cataract_model.h5')   # โมเดล 150×150×3 → 2 class

# ----------------- Mediapipe -----------------
mp_face_mesh = mp.solutions.face_mesh
mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,          # จุดรูม่านตาชัด
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

# ดัชนี landmark รอบตาซ้าย/ขวา (เลือก 4 จุดมุม) ----------------
LEFT  = [33, 133, 159, 145]
RIGHT = [362, 263, 386, 374]

# ----------------- กล้อง -----------------
cam = Picamera2()
cam.preview_configuration.main.size = (640, 480)
cam.configure("preview")
cam.start()
print("กดปุ่มสีขาวเพื่อถ่าย  |  กดปุ่มสีเหลืองเพื่อออก")
blue.on()

try:
    while True:
        frame = cam.capture_array()                 # RGB24 (PiCamera2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # -------- Mediapipe --------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mesh.process(rgb)

        eye_detected = False
        eye_crop = None

        if res.multi_face_landmarks:
            h, w = frame.shape[:2]
            lm = res.multi_face_landmarks[0].landmark

            # เลือกตาซ้าย (เปลี่ยนเป็น RIGHT ได้ตามต้องการ)
            xs = [int(lm[i].x * w) for i in LEFT]
            ys = [int(lm[i].y * h) for i in LEFT]

            pad = 5
            x1, x2 = max(0, min(xs)-pad), min(w, max(xs)+pad)
            y1, y2 = max(0, min(ys)-pad), min(h, max(ys)+pad)

            if (x2-x1) > 0 and (y2-y1) > 0:
                eye_crop = frame[y1:y2, x1:x2]
                eye_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # -------- แสดงภาพพรีวิว --------
        cv2.imshow("Camera Preview", frame)
        cv2.waitKey(1)

        # -------- ปุ่มถ่าย --------
        if button.is_pressed:
            blue.off()
            yellow.blink(on_time=0.5, off_time=0.5)

            if eye_detected and eye_crop is not None:
                # เตรียมภาพเข้าโมเดล
                img = cv2.resize(eye_crop, (150, 150))
                img_arr = image.img_to_array(img) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)

                pred = model.predict(img_arr)
                print("Prediction:", pred)

                yellow.off()

                if pred[0, 0] >= 0.1:          # ค่า threshold
                    red.on();  result = "ผลลัพธ์: Cataract"
                else:
                    green.on(); result = "ผลลัพธ์: Normal"
            else:
                result = "ไม่พบดวงตาชัดเจน"
                yellow.off(); green.blink(on_time=0.5, off_time=0.5)

            print(result)
            sleep(5)
            red.off(); green.off(); yellow.off(); blue.on()

        # -------- ปุ่มออก --------
        if exit_but.is_pressed:
            print("Exit requested.")
            break

finally:
    mesh.close()
    cam.stop()
    cv2.destroyAllWindows()
    blue.off(); red.off(); green.off(); yellow.off()
