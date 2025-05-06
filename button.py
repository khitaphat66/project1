from picamera2 import Picamera2
import cv2
import numpy as np
from time import sleep
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gpiozero import LED, Button
from signal import pause

button = Button(18)
red = LED(17)
green = LED(27)
yellow = LED(22)
blue = LED(5)

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
    
loaded_model = load_model('/boot/overlays/cataract_model.h5')
#loaded_model.summary()

camera = Picamera2()
camera.preview_configuration.main.size = (640, 480)
camera.configure("preview")
camera.start()
print("กด c เพื่อถ่ายหรือกด q เพื่อออก")
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

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        blue.off()
        
        yellow.blink(on_time = 0.5, off_time = 0.5)
        
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

    elif key == ord('q'):
        
        break

blue.off()
red.off()
green.off()
camera.stop()
cv2.destroyAllWindows()


------------------------------------------------------------------------
from gpiozero import LED, Button
from signal import pause

print("hi")

led = LED(17)
button = Button(18)
    
button.when_pressed = led.on
button.when_released = led.off

pause()


-------------------------------
from picamera2 import Picamera2
import cv2
import numpy as np
from time import sleep
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gpiozero import LED, Button
from signal import pause

button = Button(18)
red = LED(17)
green = LED(27)
yellow = LED(22)
blue = LED(5)

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
    
loaded_model = load_model('/boot/overlays/cataract_model.h5')

camera = Picamera2()
camera.preview_configuration.main.size = (640, 480)
camera.configure("preview")
camera.start()
print("กดสวิตช์สีขาวเพื่อถ่าย หรือกด 'q' เพื่อออก")
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

    # เช็คการกดปุ่ม 'q' เพื่อออก
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # เช็ครอปุ่มจาก GPIO
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

blue.off()
red.off()
green.off()
camera.stop()
cv2.destroyAllWindows()

---------------------------------------------------------------
from picamera2 import Picamera2
import cv2
import numpy as np
from time import sleep
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gpiozero import LED, Button

button = Button(18)        # สำหรับถ่ายภาพ
exit_button = Button(25)   # สำหรับออกจากโปรแกรม

red = LED(17)
green = LED(27)
yellow = LED(22)
blue = LED(5)

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

loaded_model = load_model('/boot/overlays/cataract_model.h5')

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

    # เช็คสวิตช์ออก
    if exit_button.is_pressed:
        print("ออกจากโปรแกรม")
        break

    # เช็คสวิตช์ถ่ายภาพ
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

blue.off()
red.off()
green.off()
camera.stop()
cv2.destroyAllWindows()
