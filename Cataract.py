from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

loaded_model = load_model('cataract_model.h5')
loaded_model.summary()
# โหลดโมเดลที่บันทึกไว้
loaded_model = load_model('cataract_model.h5')

# โหลดรูปภาพที่ต้องการทำนาย
img_path = 'E:\\project1\\eye\\test\\Cataract\\image_284.png'
img = image.load_img(img_path, target_size=(150, 150))

# แปลงรูปภาพเป็น array
img_array = image.img_to_array(img)

# ปรับขนาดให้เหมาะกับโมเดล (batch size 1)
img_array = np.expand_dims(img_array, axis=0)

# normalize
img_array /= 255.

# กำหนด label ที่แท้จริงของภาพนี้
# สมมติว่า 1 = Cataract, 0 = Normal
true_label = np.array([[1]])  # เพราะภาพนี้เป็น Cataract

# ประเมินผล
loss, accuracy = loaded_model.evaluate(img_array, true_label, verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# ทำนายผล
prediction = loaded_model.predict(img_array)

# แสดงผล prediction
print("Prediction values:", prediction)

# ดูผลว่าเป็น Cataract หรือ Normal
if prediction[0][1] > 0.5:
    print("ผล: เป็น Cataract")
else:
    print("ผล: เป็น Normal")


---------------------------------
from picamera2 import Picamera2
import cv2
import numpy as np
from time import sleep
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# โหลดโมเดล
loaded_model = load_model('/boot/overlays/cataract_model.h5')
loaded_model.summary()

# ตั้งกล้อง
camera = Picamera2()
camera.preview_configuration.main.size = (640, 480)
camera.configure("preview")
camera.start()

print("กดปุ่ม 'c' เพื่อถ่ายภาพและพยากรณ์, หรือกด 'q' เพื่อออก")

while True:
    frame = camera.capture_array()

    # วาดกรอบโฟกัส (ขนาด 200x200 ตรงกลาง)
    h, w, _ = frame.shape
    x1 = w//2 - 100
    y1 = h//2 - 100
    x2 = w//2 + 100
    y2 = h//2 + 100
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # สีเขียว กรอบหนา 2px

    cv2.imshow("Camera Preview", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # เมื่อกด 'c' => บันทึกภาพ
        capture = frame.copy()
        img_crop = capture[y1:y2, x1:x2]  # ตัดเฉพาะบริเวณกรอบโฟกัส

        # เตรียมภาพสำหรับโมเดล
        img = cv2.resize(img_crop, (150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # พยากรณ์
        prediction = loaded_model.predict(img_array)
        print("Prediction values:", prediction)

        if np.argmax(prediction) == 0:
            print("ผลลัพธ์: เป็น Cataract")
        else:
            print("ผลลัพธ์: เป็น Normal")

    elif key == ord('q'):
        # กด 'q' เพื่อออก
        break

# ปิดกล้อง
camera.stop()
cv2.destroyAllWindows()


_______________________
from picamera2 import Picamera2
import cv2
import numpy as np
from time import sleep
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# โหลดโมเดล
loaded_model = load_model('/boot/overlays/cataract_model.h5')
loaded_model.summary()

# ตั้งค่ากล้อง
camera = Picamera2()
camera.preview_configuration.main.size = (640, 480)
camera.configure("preview")
camera.start()

print("กดปุ่ม 'c' เพื่อถ่ายภาพและพยากรณ์, หรือกด 'q' เพื่อออก")

while True:
    frame = camera.capture_array()

    # แก้สีเพี้ยน
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # วาดกรอบโฟกัส
    h, w, _ = frame.shape
    x1 = w//2 - 100
    y1 = h//2 - 100
    x2 = w//2 + 100
    y2 = h//2 + 100
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # แสดง preview
    cv2.imshow("Camera Preview", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # กด 'c' เพื่อถ่ายภาพ
        capture = frame.copy()
        img_crop = capture[y1:y2, x1:x2]

        # ถ้าเป็น 4 ช่อง (BGRA) ให้แปลงเป็น 3 ช่อง (BGR)
        if img_crop.shape[2] == 4:
            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGRA2BGR)

        # เตรียมรูปเข้าโมเดล
        img = cv2.resize(img_crop, (150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # พยากรณ์
        prediction = loaded_model.predict(img_array)
        print("Prediction values:", prediction)

        if np.argmax(prediction) == 0:
            result_text = "ผลลัพธ์: เป็น Cataract"
        else:
            result_text = "ผลลัพธ์: เป็น Normal"

        print(result_text)

    elif key == ord('q'):
        # กด 'q' เพื่อออก
        break

# ปิดกล้อง
camera.stop()
cv2.destroyAllWindows()
