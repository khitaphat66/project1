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
