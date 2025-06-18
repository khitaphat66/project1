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
