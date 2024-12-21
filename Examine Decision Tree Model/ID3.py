import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages
import graphviz
import os
import sys
from sklearn.impute import SimpleImputer

sys.stdout.reconfigure(encoding='utf-8')
# Đọc dữ liệu
data = pd.read_csv('Data/diabetes.csv')
X = data.drop('Outcome', axis=1)  # Các đặc trưng
y = data['Outcome']  # Biến mục tiêu

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# Hàm vẽ đường cong học tập
def plot_learning_curve(estimator, title, X, y, ylim=(0.6, 1.1), cv=10):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Số Lượng Dữ Liệu Huấn Luyện")
    plt.ylabel("Độ Chính Xác")
    
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_scores = []
    test_scores = []
    
    for size in train_sizes:
        size = int(size * len(X))
        X_train_part, y_train_part = X[:size], y[:size]
        estimator.fit(X_train_part, y_train_part)
        train_scores.append(accuracy_score(y_train_part, estimator.predict(X_train_part)))
        test_scores.append(accuracy_score(y, estimator.predict(X)))
    
    plt.plot(train_sizes * len(X), train_scores, label="Độ Chính Xác Huấn Luyện")
    plt.plot(train_sizes * len(X), test_scores, label="Độ Chính Xác Kiểm Tra")
    plt.legend(loc="best")
    plt.grid()
    plt.show()
# Hàm vẽ ma trận nhầm lẫn
def plot_confusion_matrix(cm, classes, title='Ma Trận Nhầm Lẫn ID3', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Nhãn Thực')
    plt.xlabel('Nhãn Dự Đoán')
    plt.show()

# Hàm chạy mô hình Decision Tree
def run_decision_tree(X_train, y_train, X_test, y_test):
    # Huấn luyện cây quyết định với phương pháp ID3 (criterion='entropy')
    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model.fit(X_train, y_train)
    
    # Cross-validation
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    accuracy = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    mean_accuracy = accuracy.mean()
    std_accuracy = accuracy.std()

    # Dự đoán trên tập kiểm tra
    predictions = model.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, predictions)
    
    # Vẽ đường cong học tập
    plot_learning_curve(model, 'Đường cong học tập cho cây quyết định ID3', X_train, y_train)
    
    # Vẽ ma trận nhầm lẫn
    plot_confusion_matrix(cnf_matrix, classes=['Khỏe Mạnh', 'Tiểu Đường'], title='Ma Trận Nhầm Lẫn')
    
    # In kết quả
    print(f'DecisionTreeClassifier (ID3) - Độ Chính Xác Tập Huấn Luyện: {mean_accuracy:.2f} ({std_accuracy:.2f})')
    return model

# Hàm lưu cây quyết định vào PDF
def save_decision_tree_to_pdf(model, feature_names, class_names, output_file):
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render(output_file, format='pdf', cleanup=True)
    print(f"Cây quyết định đã được lưu vào: {output_file}.pdf")



# Chạy mô hình và trực quan hóa
model = run_decision_tree(X_train, y_train, X_test, y_test)

# Huấn luyện mô hình và lưu cây vào PDF
clf = DecisionTreeClassifier(
    criterion='entropy',   # Sử dụng entropy cho phương pháp ID3
    max_depth=4,           # Giới hạn độ sâu của cây
    min_samples_leaf=10,   # Số lượng mẫu tối thiểu ở mỗi lá
    random_state=42         # Đảm bảo tính tái lập
)

clf.fit(X_train, y_train)

# Lưu cây quyết định vào PDF
save_decision_tree_to_pdf(
    clf,
    feature_names=data.columns[:-1],  # Tên các đặc trưng
    class_names=["Khỏe Mạnh", "Tiểu Đường"],  # Tên các lớp
    output_file="decision_tree_ID3"  # Tên tệp đầu ra
)
# run_decision_tree(X_train, y_train, X_test, y_test)

# Hàm dự đoán bệnh tiểu đường từ mô hình đã huấn luyện
def predict_diabetes(model, input_data):
    """
    Dự đoán kết quả bệnh tiểu đường dựa trên dữ liệu đầu vào.
    """
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_data_df = pd.DataFrame([input_data], columns=columns)
    prediction = model.predict(input_data_df)

    return "Tiểu đường" if prediction == 1 else "Không có tiểu đường"

# Hàm dự đoán từ dữ liệu người dùng nhập vào
def predict_diabetes_from_input(model):
    print("Vui lòng nhập các giá trị sau:")
    
    try:
        pregnancies = int(input("Số lần mang thai (Pregnancies): "))
        glucose = float(input("Mức glucose (Glucose): "))
        blood_pressure = float(input("Huyết áp (Blood Pressure): "))
        skin_thickness = float(input("Độ dày da (Skin Thickness): "))
        insulin = float(input("Mức insulin (Insulin): "))
        bmi = float(input("Chỉ số BMI (BMI): "))
        diabetes_pedigree_function = float(input("Hệ số di truyền (Diabetes Pedigree Function): "))
        age = int(input("Tuổi (Age): "))

        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]

        # Dự đoán kết quả từ mô hình
        prediction = predict_diabetes(model, input_data)

        print(f"\nKết quả dự đoán: {prediction}")
        return prediction
    except ValueError:
        print("Vui lòng nhập dữ liệu hợp lệ!")
        return None

# In cây quyết định dưới dạng văn bản
from sklearn.tree import export_text
tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)

# Gọi hàm dự đoán từ dữ liệu người dùng nhập vào
predict_diabetes_from_input(model)