import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
import graphviz
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình lại đường dẫn tới dot.exe (cần thiết nếu bạn đã cài đặt Graphviz)

# Hàm đọc và tiền xử lý dữ liệu từ file CSV
def read_and_preprocess_data(file_path):
    """
    Đọc và tiền xử lý dữ liệu từ file CSV.
    Tiến hành mã hóa nhãn và phân chia dữ liệu thành các đặc trưng (X) và nhãn (y).

    Parameters:
    - file_path: Đường dẫn đến file CSV đã tiền xử lý

    Returns:
    - X: Các đặc trưng (features) cho mô hình
    - y: Nhãn (labels) cho mô hình
    """
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(file_path)

    # Tiền xử lý dữ liệu:
    # Mã hóa nhãn Outcome thành dạng số
    label_encoder = LabelEncoder()
    df['Outcome'] = label_encoder.fit_transform(df['Outcome'])  # 'Diabetes' -> 1, 'No Diabetes' -> 0

    # Chia dữ liệu thành X (features) và y (labels)
    X = df.drop(['Outcome'], axis=1)  # Các đặc trưng (features) (loại bỏ cột Outcome)
    y = df['Outcome']  # Nhãn (Outcome)

    return X, y


# Đọc và tiền xử lý dữ liệu từ tệp CSV
file_path = 'Data/diabetes.csv'  # Đảm bảo bạn có đường dẫn chính xác tới tệp dữ liệu
X, y = read_and_preprocess_data(file_path)

# Chia dữ liệu thành tập huấn luyện và kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Hàm huấn luyện và đánh giá mô hình Decision Tree
def run_decision_tree(X_train, y_train, X_test, y_test, max_depth=None):
    model = DecisionTreeClassifier(max_depth=max_depth)  # Tạo mô hình DecisionTree với độ sâu giới hạn
    accuracy_scorer = make_scorer(accuracy_score)  # Đo lường độ chính xác
    model.fit(X_train, y_train)  # Huấn luyện mô hình với dữ liệu huấn luyện

    kfold = KFold(n_splits=10, shuffle=True, random_state=7)  # Chia dữ liệu thành 10 phần với shuffle=True
    accuracy = cross_val_score(model, X_train, y_train, cv=kfold,
                               scoring='accuracy')  # Đánh giá mô hình bằng cross-validation
    mean_accuracy = accuracy.mean()  # Tính độ chính xác trung bình
    stdev_accuracy = accuracy.std()  # Tính độ lệch chuẩn

    # Dự đoán kết quả với dữ liệu kiểm tra
    prediction = model.predict(X_test)

    # Tính ma trận nhầm lẫn
    cnf_matrix = confusion_matrix(y_test, prediction)

    # Vẽ đường cong học (learning curve)
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=10, n_jobs=-1)

    # Tính điểm trung bình cho đường cong học
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    # Vẽ đường cong học
    plt.plot(train_sizes, train_mean, label='Độ chính xác trên tập huấn luyện')
    plt.plot(train_sizes, test_mean, label='Độ chính xác trên tập kiểm tra')
    plt.xlabel('Kích thước tập huấn luyện')
    plt.ylabel('Độ chính xác')
    plt.title('Đường cong học cho DecisionTreeClassifier')
    plt.legend(loc='best')
    plt.show()  # Hiển thị đồ thị đường cong học

    # Vẽ ma trận nhầm lẫn bằng ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=['Không có tiểu đường', 'Tiểu đường'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Ma trận nhầm lẫn')
    plt.show()  # Hiển thị đồ thị ma trận nhầm lẫn

    # In độ chính xác
    print(f'DecisionTreeClassifier - Độ chính xác trên tập huấn luyện: {mean_accuracy:.4f} (+/- {stdev_accuracy:.4f})')

    return model


# Gọi hàm run_decision_tree với dữ liệu huấn luyện và kiểm tra, giới hạn độ sâu của cây là 4
max_depth_value = 4
model = run_decision_tree(X_train, y_train, X_test, y_test, max_depth=max_depth_value)


# Hàm vẽ cây quyết định
def plot_decision_tree(model, feature_names):
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=feature_names,
                               class_names=['Không có tiểu đường', 'Tiểu đường'],  # Tên các lớp
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)  # Tạo đồ thị từ cây quyết định
    return graph


# Vẽ cây quyết định
graph = plot_decision_tree(model, X.columns)
graph.render("decision_tree_CART")  # Lưu cây quyết định vào file PDF
graph.view()  # Mở cây quyết định trong trình duyệt hoặc trình đọc PDF


# Hàm dự đoán bệnh tiểu đường từ mô hình đã huấn luyện
def predict_diabetes(model, input_data):
    """
    Dự đoán kết quả bệnh tiểu đường dựa trên dữ liệu đầu vào.

    Parameters:
    - model: Mô hình đã huấn luyện (DecisionTreeClassifier).
    - input_data: Dữ liệu đầu vào là danh sách các đặc trưng của bệnh nhân.

    Returns:
    - Kết quả dự đoán: "Tiểu đường" hoặc "Không có tiểu đường"
    """
    # Tạo DataFrame với tên các cột tương ứng với các đặc trưng mà mô hình đã huấn luyện
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_data_df = pd.DataFrame([input_data], columns=columns)

    # Dự đoán với dữ liệu đầu vào
    prediction = model.predict(input_data_df)

    # Xử lý kết quả dự đoán (0: Không có tiểu đường, 1: Tiểu đường)
    if prediction == 1:
        return "Tiểu đường"
    else:
        return "Không có tiểu đường"


def predict_diabetes_from_input(model):
    """
    Yêu cầu người dùng nhập các đặc trưng đầu vào theo thứ tự điều kiện cây quyết định
    và dự đoán kết quả bệnh tiểu đường sau khi tất cả dữ liệu được nhập.

    Parameters:
    - model: Mô hình đã huấn luyện (DecisionTreeClassifier).

    Returns:
    - Dự đoán: "Tiểu đường" hoặc "Không có tiểu đường" tùy vào kết quả dự đoán.
    """
    print("Vui lòng nhập các giá trị sau:")

    # Yêu cầu người dùng nhập các đặc trưng theo thứ tự điều kiện trong cây quyết định
    pregnancies = int(input("Số lần mang thai (Pregnancies): "))
    glucose = float(input("Mức glucose (Glucose): "))
    blood_pressure = float(input("Huyết áp (Blood Pressure): "))
    skin_thickness = float(input("Độ dày da (Skin Thickness): "))
    insulin = float(input("Mức insulin (Insulin): "))
    bmi = float(input("Chỉ số BMI (BMI): "))
    diabetes_pedigree_function = float(input("Hệ số di truyền (Diabetes Pedigree Function): "))
    age = int(input("Tuổi (Age): "))

    # Tạo danh sách đặc trưng từ dữ liệu nhập vào
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]

    # Sau khi tất cả dữ liệu được nhập, dự đoán kết quả từ mô hình
    prediction = predict_diabetes(model, input_data)

    # In kết quả dự đoán
    print(f"\nKết quả dự đoán: {prediction}")
    return prediction

from sklearn.tree import export_text

# In cây quyết định dưới dạng văn bản
tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)

# Gọi hàm dự đoán từ dữ liệu người dùng nhập vào
predict_diabetes_from_input(model)


