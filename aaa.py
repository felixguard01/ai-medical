# Import libraries 
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load data
file_path = r'C:\dev\Research\Data Frame\csv-data\merged_data_fix1.csv'
data = pd.read_csv(file_path)

# Quick overview of the data
data.head()

# Remove unnecessary columns
X = data.drop(columns=['filename', 'gender', 'age', 'Normal/Disorder', 
                       'S1a_1c_LA', 'S1a_1r_LA', 'S1a_2c_LA', 'S1a_2r_LA'])

# แยกฟีเจอร์ที่เกี่ยวข้องกับ AP
X_ap_features = [col for col in data.columns if '_AP' in col]
X_ap = data[X_ap_features]

# Save filtered data to CSV
output_csv = 'filtered_ap_features.csv'
X_ap.to_csv(output_csv, index=False)
print(f"บันทึกข้อมูลเป็นไฟล์ CSV: {output_csv}")

# เป้าหมาย AP
y_ap = data[['L1b_d_AP', 'L2b_d_AP', 'L3b_d_AP', 'L4b_d_AP', 'L5b_d_AP']]

# ฟังก์ชันสำหรับกรองเป้าหมาย
def filter_targets(y, valid_targets=[0, 1, 2, 3, 4]):
    return y.apply(lambda col: col.where(col.isin(valid_targets))).dropna()

# กรองเป้าหมาย AP
y_ap_filtered = filter_targets(y_ap)
X_filtered_ap = X_ap.loc[y_ap_filtered.index]

# Split data
X_train_ap, X_test_ap, y_train_ap, y_test_ap = train_test_split(
    X_filtered_ap, y_ap_filtered, test_size=0.2, random_state=42
)

# Scaling features
scaler_ap = StandardScaler()
X_train_ap = scaler_ap.fit_transform(X_train_ap)
X_test_ap = scaler_ap.transform(X_test_ap)

# โมเดล MLP
mlp_ap = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, verbose=True)

# ใช้ MultiOutputClassifier เพื่อฝึกหลายเป้าหมาย
multi_target_mlp_ap = MultiOutputClassifier(mlp_ap)
multi_target_mlp_ap.fit(X_train_ap, y_train_ap)



# Save the model
folder_name = 'ap-model'
os.makedirs(folder_name, exist_ok=True)
model_filename = os.path.join(folder_name, 'mlp_ap_model.pkl')
joblib.dump(multi_target_mlp_ap, model_filename)
print(f"โมเดลถูกบันทึกเป็นไฟล์ในโฟลเดอร์: {model_filename}")

# Predict และ Evaluate
y_pred_ap = multi_target_mlp_ap.predict(X_test_ap)

print("Classification Reports for Filtered AP Targets:")
for i, column in enumerate(y_ap_filtered.columns):
    print(f"\nTarget: {column}")
    print(classification_report(y_test_ap.iloc[:, i], y_pred_ap[:, i]))

# Overall Classification Report
y_ap_true_flat = y_test_ap.values.flatten()
y_ap_pred_flat = y_pred_ap.flatten()
print("\nOverall Filtered AP Classification Report:")
print(classification_report(y_ap_true_flat, y_ap_pred_flat, zero_division=1))

overall_accuracy_ap = sum(accuracy_score(y_test_ap.iloc[:, i], y_pred_ap[:, i]) for i in range(y_ap_filtered.shape[1])) / y_ap_filtered.shape[1]
print(f"\nOverall Filtered AP Accuracy Across All Targets: {overall_accuracy_ap}")

# สร้างกราฟ Loss Curve
if hasattr(mlp_ap, "loss_curve_") and len(mlp_ap.loss_curve_) > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(mlp_ap.loss_curve_, label='Training Loss')
    plt.title('Training Loss Curve for AP MLP')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No loss curve is available.")
