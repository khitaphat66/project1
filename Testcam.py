import cv2
import numpy as np
from tensorflow.keras.models import load_model

# โหลดโมเดล .h5
model = load_model('your_model.h5')  # ใส่ชื่อไฟล์โมเดลของคุณตรงนี้

# กำหนดขนาด input ของโมเดล (ปรับให้ตรงกับตอนเทรน เช่น 224x224 หรือ 128x128)
IMG_SIZE = (224, 224)

# เปิดกล้อง
cap = cv2.VideoCapture(0)  # 0 = default camera

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

while True:
    # อ่านเฟรมจากกล้อง
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถอ่านภาพจากกล้องได้")
        break

    # แสดงภาพสด
    cv2.imshow('Camera', frame)

    # กด 'c' เพื่อ capture ภาพไปทำนาย
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # เตรียมภาพก่อนส่งเข้าโมเดล
        img = cv2.resize(frame, IMG_SIZE)
        img = img / 255.0  # ปรับค่าสีให้อยู่ในช่วง 0-1 ถ้าโมเดลเทรนแบบ normalize มา
        img = np.expand_dims(img, axis=0)  # เพิ่มมิติให้กลายเป็น (1, IMG_SIZE[0], IMG_SIZE[1], 3)

        # ทำนาย
        prediction = model.predict(img)
        # สมมุติ output เป็น 1 ค่า คือ ความน่าจะเป็น
        prob = prediction[0][0]
        print(f"โอกาสเป็นต้อกระจก: {prob:.2f}")

        # ตัดสินใจ
        if prob > 0.5:
            print("=> เป็นต้อกระจก")
        else:
            print("=> ไม่เป็นต้อกระจก")

    # กด 'q' เพื่อออกจาก loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()



---------------------
from picamera import PiCamera
from time import sleep
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite

# ตั้งค่ากล้อง
camera = PiCamera()
camera.resolution = (224, 224)  # ขนาดภาพเล็กลง เพื่อส่งเข้าโมเดล

# ถ่ายภาพตา
image_path = '/home/pi/eye_image.jpg'
print("กำลังถ่ายภาพ...")
camera.start_preview()
sleep(2)  # รอให้กล้องโฟกัส
camera.capture(image_path)
camera.stop_preview()
print("ถ่ายเสร็จแล้ว บันทึกที่", image_path)


-------------------------     
from picamera2 import Picamera2, Preview
from time import sleep
import cv2
import numpy as np

# ตั้งค่ากล้อง
camera = Picamera2()
camera.resolution = (224, 224)  # ขนาดภาพเล็กลง เพื่อส่งเข้าโมเดล

# ถ่ายภาพ
image_path = '/home/user/Downloads/eye_image.jpg'
print("กำลังถ่ายภาพ...")

camera.start_preview(Preview.QTGL)  # แสดง Preview
camera.start()
sleep(2)  # รอให้กล้องโฟกัส
camera.capture_file(image_path)  # ถ่ายรูปแล้วบันทึก
camera.stop_preview()

print(f"ถ่ายเสร็จแล้ว บันทึกที่: {image_path}")

# วิเคราะภาพ: ใช้ cv2 และ numpy
img = cv2.imread(image_path)  # ใช้ OpenCV อ่านภาพ
print("ขนาดรูป:", img.shape)  # ควรเป็น (224, 224, 3)

# ตัวอย่างการแปลงเป็น grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("ขนาดภาพ grayscale:", gray_img.shape)  # ควรเป็น (224, 224)

# คุณสามารถใส่โค้ดวิเคราะห์หรือโมเดลที่นี่ เช่นส่งเข้าโมเดล ML



---------------------
from picamera2 import Picamera2, Preview
from time import sleep
import cv2
import numpy as np

# ตั้งค่ากล้อง
camera = Picamera2()
camera.resolution = (224, 224)  # ขนาดภาพเล็กลง เพื่อส่งเข้าโมเดล

# ตรวจสอบการตั้งค่ากล้อง
print(f"ตั้งค่ากล้องเป็น: {camera.resolution}")

# ถ่ายภาพ
image_path = '/home/user/Downloads/eye_image.jpg'
print("กำลังถ่ายภาพ...")

camera.start_preview(Preview.QTGL)  # แสดง Preview
camera.start()
sleep(2)  # รอให้กล้องโฟกัส
camera.capture_file(image_path)  # ถ่ายรูปแล้วบันทึก
camera.stop_preview()

# อ่านภาพที่ถ่ายมา
img = cv2.imread(image_path)  # ใช้ OpenCV อ่านภาพ
print("ขนาดรูปก่อนปรับขนาด:", img.shape)  # ดูขนาดภาพที่ถ่ายมา

# ปรับขนาดภาพให้เป็น 224x224
if img.shape[0] != 224 or img.shape[1] != 224:
    img_resized = cv2.resize(img, (224, 224))
    print("ขนาดรูปหลังปรับขนาด:", img_resized.shape)  # ควรเป็น (224, 224, 3)
else:
    img_resized = img
    print("ขนาดรูปไม่ต้องปรับขนาด (แล้วแต่กล้อง):", img_resized.shape)

# ตัวอย่างการแปลงเป็น grayscale
gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
print("ขนาดภาพ grayscale:", gray_img.shape)  # ควรเป็น (224, 224)
