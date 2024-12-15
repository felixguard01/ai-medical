#แยก AP และ LA
#พิจารณาเพิ่มตัวอย่างในกลุ่มเป้าหมายเล็ก (เช่น 3 และ 4) เพื่อปรับปรุง Recall
#สามารถใช้วิธี Oversampling หรือ Data Augmentation เพื่อปรับปรุงความสมดุลของข้อมูล

#2. ปัญหาที่พบ
#ข้อความ Warning:

#UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
#สาเหตุ: ในกลุ่มเป้าหมายเล็ก (เช่น กลุ่ม 3 และ 4) ไม่มีการทำนายตัวอย่างที่ถูกต้อง หรือโมเดลไม่ได้ทำนายค่าเหล่านี้เลย
#วิธีแก้ไข:
#ใช้พารามิเตอร์ zero_division=1 ใน classification_report (ซึ่งคุณได้ทำแล้ว)
#เพิ่มตัวอย่างในกลุ่มเป้าหมายเล็กเพื่อช่วยให้โมเดลเรียนรู้ดีขึ้น (อาจพิจารณาการทำ Oversampling เช่น SMOTE)

#กราฟ loss validation
#check input > part ไหน ข้อมูลเข้าไปตัว model
#cross validation 
#model deep learning > 5 อัน เหมาะกับ data เรา > สามารถปรับ พรารามิเตอร์แต่ละตัวดีกว่า ML ปกติ แต่ layer  เหมือนกันเอา
#accuracy > confustion matrix
#data prep make sence ได้แล้ว >> ด่วน พฤ
#ทยอยทำถามตาบ


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
# 1. Load the data
file_path = r'C:\dev\Research\Data Frame\csv-data\merged_data_fix1.csv'
data = pd.read_csv(file_path)

# ลบคอลัมน์ที่ไม่ต้องการจาก X
X = data.drop(columns=['filename', 'gender', 'age', 'Normal/Disorder', 
                       'S1a_1c_LA', 'S1a_1r_LA', 'S1a_2c_LA', 'S1a_2r_LA'])

# แยกฟีเจอร์สำหรับ AP และ LA
X_ap_features = [col for col in X.columns if '_AP' in col]  # ฟีเจอร์ที่เกี่ยวข้องกับ AP
X_la_features = [col for col in X.columns if '_LA' in col]  # ฟีเจอร์ที่เกี่ยวข้องกับ LA

# สร้าง X_ap และ X_la
X_ap = X[X_ap_features]  # ฟีเจอร์สำหรับ AP
X_la = X[X_la_features]  # ฟีเจอร์สำหรับ LA


y_ap = data[['L1b_d_AP', 'L2b_d_AP', 'L3b_d_AP', 'L4b_d_AP', 'L5b_d_AP']]  # AP Targets
y_la = data[['L1b_d_LA', 'L2b_d_LA', 'L3b_d_LA', 'L4b_d_LA', 'L5b_d_LA']]  # LA Targets

# ฟังก์ชันสำหรับกรองเป้าหมาย
def filter_targets(y, valid_targets=[0, 1, 2, 3, 4]):
    return y.apply(lambda col: col.where(col.isin(valid_targets))).dropna()

# กรองเป้าหมาย AP และ LA
y_ap_filtered = filter_targets(y_ap)
y_la_filtered = filter_targets(y_la)

# กรอง X เพื่อให้สอดคล้องกับแถวที่เหลือใน y_ap และ y_la หลังการกรอง
X_filtered_ap = X.loc[y_ap_filtered.index]
X_filtered_la = X.loc[y_la_filtered.index]

# Split data ใหม่หลังการกรอง
X_train_ap, X_test_ap, y_train_ap, y_test_ap = train_test_split(
    X_filtered_ap, y_ap_filtered, test_size=0.2, random_state=42
)

X_train_la, X_test_la, y_train_la, y_test_la = train_test_split(
    X_filtered_la, y_la_filtered, test_size=0.2, random_state=42
)

# Scaling features สำหรับ AP และ LA
scaler_ap = StandardScaler()
X_train_ap = scaler_ap.fit_transform(X_train_ap)
X_test_ap = scaler_ap.transform(X_test_ap)

scaler_la = StandardScaler()
X_train_la = scaler_la.fit_transform(X_train_la)
X_test_la = scaler_la.transform(X_test_la)

# โมเดล MLP
mlp_ap = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
mlp_la = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)

# Train โมเดล AP
multi_target_mlp_ap = MultiOutputClassifier(mlp_ap)
multi_target_mlp_ap.fit(X_train_ap, y_train_ap)

# Train โมเดล LA
multi_target_mlp_la = MultiOutputClassifier(mlp_la)
multi_target_mlp_la.fit(X_train_la, y_train_la)

# Predict ผลลัพธ์
y_pred_ap = multi_target_mlp_ap.predict(X_test_ap)
y_pred_la = multi_target_mlp_la.predict(X_test_la)

# Evaluate โมเดลสำหรับ AP
print("Classification Reports for Filtered AP Targets:")
for i, column in enumerate(y_ap_filtered.columns):
    print(f"\nTarget: {column}")
    print(classification_report(y_test_ap.iloc[:, i], y_pred_ap[:, i]))

# Evaluate โมเดลสำหรับ LA
print("\nClassification Reports for Filtered LA Targets:")
for i, column in enumerate(y_la_filtered.columns):
    print(f"\nTarget: {column}")
    print(classification_report(y_test_la.iloc[:, i], y_pred_la[:, i]))

# Overall Classification Report สำหรับ AP
y_ap_true_flat = y_test_ap.values.flatten()
y_ap_pred_flat = y_pred_ap.flatten()
print("\nOverall Filtered AP Classification Report:")
print(classification_report(y_ap_true_flat, y_ap_pred_flat, zero_division=1))

# Overall Classification Report สำหรับ LA
y_la_true_flat = y_test_la.values.flatten()
y_la_pred_flat = y_pred_la.flatten()
print("\nOverall Filtered LA Classification Report:")
print(classification_report(y_la_true_flat, y_la_pred_flat, zero_division=1))

# Overall Accuracy สำหรับ AP
overall_accuracy_ap = sum(accuracy_score(y_test_ap.iloc[:, i], y_pred_ap[:, i]) for i in range(y_ap_filtered.shape[1])) / y_ap_filtered.shape[1]
print(f"\nOverall Filtered AP Accuracy Across All Targets: {overall_accuracy_ap}")

# Overall Accuracy สำหรับ LA
overall_accuracy_la = sum(accuracy_score(y_test_la.iloc[:, i], y_pred_la[:, i]) for i in range(y_la_filtered.shape[1])) / y_la_filtered.shape[1]
print(f"\nOverall Filtered LA Accuracy Across All Targets: {overall_accuracy_la}")


# Random Forest parameters
rf_ap = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_la = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)

# Train Random Forest model for AP
multi_target_rf_ap = MultiOutputClassifier(rf_ap)
multi_target_rf_ap.fit(X_train_ap, y_train_ap)

# Train Random Forest model for LA
multi_target_rf_la = MultiOutputClassifier(rf_la)
multi_target_rf_la.fit(X_train_la, y_train_la)

# Predict results
y_pred_ap_rf = multi_target_rf_ap.predict(X_test_ap)
y_pred_la_rf = multi_target_rf_la.predict(X_test_la)

# Evaluate model for AP
print("Classification Reports for Filtered AP Targets (Random Forest):")
for i, column in enumerate(y_ap_filtered.columns):
    print(f"\nTarget: {column}")
    print(classification_report(y_test_ap.iloc[:, i], y_pred_ap_rf[:, i]))

# Evaluate model for LA
print("\nClassification Reports for Filtered LA Targets (Random Forest):")
for i, column in enumerate(y_la_filtered.columns):
    print(f"\nTarget: {column}")
    print(classification_report(y_test_la.iloc[:, i], y_pred_la_rf[:, i]))

# Overall Classification Report for AP
y_ap_rf_true_flat = y_test_ap.values.flatten()
y_ap_rf_pred_flat = y_pred_ap_rf.flatten()
print("\nOverall Filtered AP Classification Report (Random Forest):")
print(classification_report(y_ap_rf_true_flat, y_ap_rf_pred_flat, zero_division=1))

# Overall Classification Report for LA
y_la_rf_true_flat = y_test_la.values.flatten()
y_la_rf_pred_flat = y_pred_la_rf.flatten()
print("\nOverall Filtered LA Classification Report (Random Forest):")
print(classification_report(y_la_rf_true_flat, y_la_rf_pred_flat, zero_division=1))

# Overall Accuracy for AP
overall_accuracy_ap_rf = sum(accuracy_score(y_test_ap.iloc[:, i], y_pred_ap_rf[:, i]) for i in range(y_ap_filtered.shape[1])) / y_ap_filtered.shape[1]
print(f"\nOverall Filtered AP Accuracy Across All Targets (Random Forest): {overall_accuracy_ap_rf}")

# Overall Accuracy for LA
overall_accuracy_la_rf = sum(accuracy_score(y_test_la.iloc[:, i], y_pred_la_rf[:, i]) for i in range(y_la_filtered.shape[1])) / y_la_filtered.shape[1]
print(f"\nOverall Filtered LA Accuracy Across All Targets (Random Forest): {overall_accuracy_la_rf}")