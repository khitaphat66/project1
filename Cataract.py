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
