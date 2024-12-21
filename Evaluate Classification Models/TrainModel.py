import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from collections import Counter
from imblearn.over_sampling import SMOTE  # Thêm thư viện SMOTE để xử lý mất cân bằng
from sklearn.utils.class_weight import compute_class_weight  # Tính trọng số lớp

# Đọc dữ liệu
data = pd.read_csv('Data/cleaned_diabetes.csv')
X = data.drop('Outcome', axis=1)  # Các đặc trưng
y = data['Outcome']  # Biến mục tiêu

# Kiểm tra tỷ lệ lớp ban đầu
print("Tỷ lệ lớp ban đầu:", Counter(y))

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Áp dụng SMOTE để tạo dữ liệu cho lớp ít mẫu
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Tính trọng số lớp để xử lý mất cân bằng trong mô hình
classes = np.array([0, 1])  # Chuyển đổi thành numpy array
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

# Hàm đánh giá mô hình
def evaluate_model(model, X_train, y_train, X_test, y_test, method_name, class_weight_dict=None):
    if class_weight_dict and not isinstance(model, KNeighborsClassifier):  # Không áp dụng class_weight cho KNN
        model.set_params(class_weight=class_weight_dict)  # Cập nhật trọng số lớp
    model.fit(X_train, y_train)  # Huấn luyện mô hình
    y_pred = model.predict(X_test)  # Dự đoán
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nKết quả với {method_name}:")
    print(f"Độ chính xác: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Vẽ Confusion Matrix
    plot_conf_matrix(model, X_test, y_test)
    
    # Vẽ ROC Curve
    plot_roc_curve(model, X_test, y_test)
    
    # Vẽ biểu đồ scatter
    plot_scatter(model, X_test, y_test, method_name)
    # Sau khi huấn luyện mô hình, gọi hàm vẽ biểu đồ scatter
    plot_scatter_bmi_glucose(model, X_test, y_test, method_name)

    return accuracy  # Trả về độ chính xác của mô hình

# Vẽ Confusion Matrix
def plot_conf_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Khỏe mạnh", "Tiểu đường"], yticklabels=["Không mắc bệnh", "Mắc bệnh"])
    plt.title(f'Matrix nhầm lẫn - {model.__class__.__name__}')
    plt.ylabel('Nhãn thực tế')
    plt.xlabel('Nhãn dự đoán')
    plt.show()

# Vẽ ROC Curve
def plot_roc_curve(model, X_test, y_test):
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Lấy xác suất cho class "1" (Diabetes)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Đường cong ROC (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỷ lệ dương tính giả')
    plt.ylabel('Tỷ lệ dương tính thật')
    plt.title(f'Đường cong ROC - {model.__class__.__name__}')
    plt.legend(loc="lower right")
    plt.show()

# Vẽ biểu đồ scatter cho kết quả dự đoán
def plot_scatter(model, X_test, y_test, method_name):
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_test['Glucose'], X_test['Age'], c=y_prob if y_prob is not None else y_pred, cmap='coolwarm', alpha=0.7, edgecolor='k')
    plt.colorbar(scatter, label='Xác suất dự đoán bệnh tiểu đường' if y_prob is not None else 'Nhãn dự đoán')
    plt.title(f"{method_name}: Mối quan hệ giữa Glucose và Tuổi với bệnh tiểu đường", fontsize=14)
    plt.xlabel('Glucose (Đường huyết)', fontsize=12)
    plt.ylabel('Age (Tuổi)', fontsize=12)
    plt.grid(True)
    plt.show()
def plot_scatter_bmi_glucose(model, X_test, y_test, method_name):
    # Lấy xác suất dự đoán cho lớp "1" (xác suất mắc bệnh tiểu đường)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_test['Glucose'], 
        X_test['BMI'], 
        c=y_prob if y_prob is not None else y_pred, 
        cmap='coolwarm', 
        alpha=0.7, 
        edgecolor='k'
    )
    
    # Thêm colorbar với nhãn
    plt.colorbar(scatter, label='Xác suất dự đoán bệnh tiểu đường' if y_prob is not None else 'Nhãn dự đoán')
    
    # Tùy chỉnh tiêu đề và nhãn cho các trục
    plt.title(f"{method_name}: Mối quan hệ giữa Glucose và BMI với bệnh tiểu đường", fontsize=14)
    plt.xlabel('Glucose (Đường huyết)', fontsize=12)
    plt.ylabel('BMI (Chỉ số khối cơ thể)', fontsize=12)
    
    # Hiển thị lưới và biểu đồ
    plt.grid(True)
    plt.show()

# Mô hình Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg_accuracy = evaluate_model(log_reg, X_train_resampled, y_train_resampled, X_test, y_test, "Logistic Regression", class_weight_dict)

# Mô hình Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_accuracy = evaluate_model(dt_model, X_train_resampled, y_train_resampled, X_test, y_test, "Decision Tree", class_weight_dict)

# Mô hình Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_accuracy = evaluate_model(rf_model, X_train_resampled, y_train_resampled, X_test, y_test, "Random Forest", class_weight_dict)

# Mô hình K-Nearest Neighbors
knn_model = KNeighborsClassifier()
knn_accuracy = evaluate_model(knn_model, X_train_resampled, y_train_resampled, X_test, y_test, "K-Nearest Neighbors", None)  # Không cần class_weight cho KNN

# Mô hình Support Vector Classifier
svc_model = SVC(random_state=42, probability=True)
svc_accuracy = evaluate_model(svc_model, X_train_resampled, y_train_resampled, X_test, y_test, "Support Vector Classifier", class_weight_dict)

# Vẽ biểu đồ cột so sánh độ chính xác
accuracies = [log_reg_accuracy, dt_accuracy, rf_accuracy, knn_accuracy, svc_accuracy]
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'K-Nearest Neighbors', 'SVC']

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color='skyblue')
plt.title('So sánh độ chính xác của các mô hình')
plt.xlabel('Mô hình')
plt.ylabel('Độ chính xác')
plt.ylim(0, 1)
plt.show()


