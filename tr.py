import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd  # สมมติคุณใช้ pandas เพื่ออ่านข้อมูล

# อ่านข้อมูลจากไฟล์ CSV
data = pd.read_csv("your_data.csv") 

# แยกข้อมูล features และ target
x_train = data[['feature1', 'feature2', ...]] 
y_train = data['target'] 

# ... (ส่วนที่เหลือของโค้ดของคุณ)

model = Sequential([
    Dense(units=64, activation='relu', input_shape=(784,)),
    Dense(units=10, activation='softmax')
])

# ... (ส่วนที่เหลือของโค้ดของคุณ)

model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), callbacks=[early_stop])