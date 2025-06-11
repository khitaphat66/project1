from picamera2 import Picamera2
import cv2
import numpy as np
from time import sleep
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gpiozero import LED, Button
from signal import pause

# ปุ่มและไฟ LED
button = Button(18)
red = LED(17)
green = LED(27)
yellow = LED(22)
blue = LED(5)

# กระพริบไฟ 4 ครั้งตอนเริ่ม
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
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# ตั้งค่ากล้อง
camera = Picamera2()
camera.preview_configuration.main.size = (640, 480)
camera.configure("preview")
camera.start()
print("กด c เพื่อถ่ายหรือกด q เพื่อออก")
blue.on()

while True:
    frame = camera.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # แปลงเป็นภาพ grayscale เพื่อหา eye
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ตรวจจับดวงตา
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # วาดกรอบรอบดวงตาที่เจอ
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
    
    cv2.imshow("Camera Preview", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('c') and len(eyes) > 0:
        blue.off()
        yellow.blink(on_time=0.5, off_time=0.5)
        
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
    
    elif key == ord('q'):
        break

# ปิดไฟทั้งหมด ปิดกล้อง
blue.off()
red.off()
green.off()
camera.stop()
cv2.destroyAllWindows()
