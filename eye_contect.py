import cv2

# โหลดโมเดล Haar cascade สำหรับใบหน้าและดวงตา
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# เปิดกล้อง (0 หมายถึงกล้องเว็บแคมตัวหลัก)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # อ่านภาพจากกล้อง
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # แปลงภาพเป็น grayscale

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # ตรวจจับใบหน้า
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)  # ตรวจจับดวงตา
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)  # วาดกรอบรอบตา

    cv2.imshow('Eye Detection', frame)  # แสดงผลภาพ

    if cv2.waitKey(1) & 0xFF == ord('q'):  # กด 'q' เพื่อออก
        break

cap.release()
cv2.destroyAllWindows()
