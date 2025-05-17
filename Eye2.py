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
camera.set_controls({"AwbEnable": False})  # ปิดการปรับสมดุลสีอัตโนมัติ
camera.set_controls({"AwbRedGain": 1.0, "AwbBlueGain": 1.0})  # ปรับสมดุลสีมือ

print("กดปุ่มสีขาวเพื่อถ่าย หรือปุ่มสีเหลืองเพื่อออก")
blue.on()

# ฟังก์ชันกรองดวงตา
def is_valid_eye(w, h):
    aspect_ratio = w / h
    return 0.3 < aspect_ratio < 3.5 and 5 < w < 200 and 5 < h < 200

while True:
    frame = camera.capture_array("main")
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # แปลงจาก RGB เป็น BGR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้า + ดวงตา
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eye_detected = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            if ey + eh < h // 2 and is_valid_eye(ew, eh):
                eye_detected = True
                break

    # วาดกรอบกลางภาพ
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

        if eye_detected:
            # ส่งเข้าโมเดล
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
        else:
            result_text = "ไม่พบตาจริง: แสดงผลเป็น Normal"
            print("No valid eyes detected, skipping model.")
            yellow.off()
            green.on()

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

----------------------------------------------------------------
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
camera.set_controls({"AwbEnable": False})  # ปิดการปรับสมดุลสีอัตโนมัติ
camera.set_controls({"AwbRedGain": 1.0, "AwbBlueGain": 1.0})  # ปรับสมดุลสีมือ

print("กดปุ่มสีขาวเพื่อถ่าย หรือปุ่มสีเหลืองเพื่อออก")
blue.on()

# ฟังก์ชันกรองดวงตา
def is_valid_eye(w, h):
    aspect_ratio = w / h
    return 0.3 < aspect_ratio < 3.5 and 5 < w < 200 and 5 < h < 200

while True:
    frame = camera.capture_array("main")
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # แปลงจาก RGB เป็น BGR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้า + ดวงตา
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eye_detected = False
    eye_frame = None

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            if ey + eh < h // 2 and is_valid_eye(ew, eh):
                eye_detected = True
                eye_frame = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]  # เลือกกรอบดวงตาที่พบ
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)  # วาดกรอบสีเขียว
                break

    cv2.imshow("Face & Eye Detection + Cataract Classification", frame)
    cv2.waitKey(1)

    if button.is_pressed:
        blue.off()
        yellow.blink(on_time=0.5, off_time=0.5)

        if eye_frame is not None:
            # ส่งเข้าโมเดล
            img = cv2.resize(eye_frame, (150, 150))
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
        else:
            result_text = "ไม่พบตาจริง: แสดงผลเป็น Normal"
            print("No valid eyes detected, skipping model.")
            yellow.off()
            green.on()

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


----------------------------
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
camera.set_controls({"AwbEnable": False})
camera.set_controls({"AwbRedGain": 1.0, "AwbBlueGain": 1.0})

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
    eye_detected = False
    eye_frames = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

        valid_eyes = []
        for (ex, ey, ew, eh) in eyes:
            if ey + eh < h // 2 and is_valid_eye(ew, eh):
                eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                valid_eyes.append((ex + x, ey + y, ew, eh, eye_img))  # พิกัดจริง

        if len(valid_eyes) >= 2:
            valid_eyes = sorted(valid_eyes, key=lambda e: e[0])[:2]
            eye_labels = ["ตาซ้าย", "ตาขวา"] if valid_eyes[0][0] < valid_eyes[1][0] else ["ตาขวา", "ตาซ้าย"]
            for i, (ex, ey, ew, eh, eye_img) in enumerate(valid_eyes):
                cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                cv2.putText(frame, eye_labels[i], (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                eye_frames.append(eye_img)
            eye_detected = True

        elif len(valid_eyes) == 1:
            (ex, ey, ew, eh, eye_img) = valid_eyes[0]
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.putText(frame, "ตา", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            eye_frames.append(eye_img)
            eye_detected = True

    cv2.imshow("Face & Eye Detection + Cataract Classification", frame)
    cv2.waitKey(1)

    if button.is_pressed:
        blue.off()
        yellow.blink(on_time=0.5, off_time=0.5)

        if eye_detected and eye_frames:
            cataract_found = False
            for ef in eye_frames:
                img = cv2.resize(ef, (150, 150))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                prediction = loaded_model.predict(img_array)
                print("Prediction values:", prediction)

                if np.argmax(prediction) == 0:
                    cataract_found = True

            yellow.off()

            if cataract_found:
                result_text = "ผลลัพธ์: พบ Cataract อย่างน้อย 1 ข้าง"
                red.on()
            else:
                result_text = "ผลลัพธ์: ทั้งสองตาปกติ"
                green.on()
        else:
            result_text = "ไม่พบตาชัดเจน: แสดงผลเป็น Normal"
            print("No valid eyes detected, skipping model.")
            yellow.off()
            green.on()

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
-------------------------------
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
blue = LED(25)

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
camera.set_controls({"AwbEnable": True})  # ปิดการปรับสมดุลสีอัตโนมัติ

print("กดปุ่มสีขาวเพื่อถ่าย หรือปุ่มสีเหลืองเพื่อออก")
blue.on()

# ฟังก์ชันกรองดวงตา
def is_valid_eye(w, h):
    aspect_ratio = w / h
    return 0.3 < aspect_ratio < 3.5 and 5 < w < 200 and 5 < h < 200

while True:
    frame = camera.capture_array("main")
    frame = np.ascontiguousarray(frame[:, :, :3])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้า + ดวงตา
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eye_detected = False
    eye_frame = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

        valid_eyes = []
        for (ex, ey, ew, eh) in eyes:
            if ey + eh < h // 2 and is_valid_eye(ew, eh):
                eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                valid_eyes.append((ex + x, ey + y, ew, eh, eye_img))  # พิกัดจริง

        if len(valid_eyes) >= 2:
            valid_eyes = sorted(valid_eyes, key=lambda e: e[0])[:2]
            eye_labels = ["left", "right"] if valid_eyes[0][0] < valid_eyes[1][0] else ["right", "left"]
            for i, (ex, ey, ew, eh, eye_img) in enumerate(valid_eyes):
                cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                cv2.putText(frame, eye_labels[i], (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                eye_frame.append(eye_img)
            eye_detected = True

        elif len(valid_eyes) == 1:
            (ex, ey, ew, eh, eye_img) = valid_eyes[0]
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.putText(frame, "eye", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            eye_frame.append(eye_img)
            eye_detected = True

    cv2.imshow("Face & Eye Detection + Cataract Classification", frame)
    cv2.waitKey(1)

    if button.is_pressed:
        blue.off()
        yellow.blink(on_time=0.5, off_time=0.5)

        if eye_detected and eye_frame:
            cataract_found = False
            for ef in eye_frame:
                img = cv2.resize(ef, (150, 150))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                prediction = loaded_model.predict(img_array)
                print("Prediction values:", prediction)

                if np.argmax(prediction) == 0:
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


