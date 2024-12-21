import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import graphviz
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Hàm tính Gain Ratio (C4.5)
def calculate_gain_ratio(X, y):
    """ Tính Gain Ratio cho các thuộc tính """
    # Đầu tiên tính Entropy
    total_entropy = entropy(y)
    
    # Tính entropy cho mỗi thuộc tính
    gain_ratios = {}
    for feature in X.columns:
        feature_values = X[feature].unique()
        weighted_entropy = 0
        for value in feature_values:
            subset_y = y[X[feature] == value]
            weighted_entropy += (len(subset_y) / len(X)) * entropy(subset_y)
        
        # Tính Gain Ratio
        gain = total_entropy - weighted_entropy
        split_info = entropy(X[feature])
        gain_ratio = gain / split_info if split_info != 0 else 0
        gain_ratios[feature] = gain_ratio
    
    return gain_ratios

def entropy(y):
    """ Tính Entropy """
    probs = y.value_counts(normalize=True)
    return -sum(probs * np.log2(probs))

# Hàm vẽ ma trận nhầm lẫn
def plot_confusion_matrix(cm, classes, title='Ma trận nhầm lẫn C4.5', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Nhãn thực tế')
    plt.xlabel('Nhãn dự đoán')
    plt.show()

# Hàm vẽ đường cong học tập
def plot_learning_curve(estimator, title, X, y, ylim=(0.6, 1.1), cv=10):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Số lượng mẫu huấn luyện")
    plt.ylabel("Độ chính xác")
    
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_scores = []
    test_scores = []
    
    for size in train_sizes:
        size = int(size * len(X))
        X_train_part, y_train_part = X[:size], y[:size]
        estimator.fit(X_train_part, y_train_part)
        train_scores.append(accuracy_score(y_train_part, estimator.predict(X_train_part)))
        test_scores.append(accuracy_score(y, estimator.predict(X)))
    
    plt.plot(train_sizes * len(X), train_scores, label="Độ chính xác huấn luyện")
    plt.plot(train_sizes * len(X), test_scores, label="Độ chính xác kiểm tra")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

# Hàm chạy mô hình Decision Tree với C4.5
def run_decision_tree(X_train, y_train, X_test, y_test):
    # Tính Gain Ratio để chọn thuộc tính phân chia
    gain_ratios = calculate_gain_ratio(X_train, y_train)
    print(f"Gain Ratios cho các thuộc tính: {gain_ratios}")
    
    # Huấn luyện cây quyết định với phương pháp C4.5
    model = DecisionTreeClassifier(
        criterion='entropy',  # C4.5 sử dụng entropy
        splitter='best',      # Phương pháp phân chia tốt nhất
        max_depth=4,          # Giới hạn độ sâu của cây
        min_samples_leaf=10,  # Số lượng mẫu tối thiểu ở mỗi lá
        random_state=42       # Đảm bảo tính tái lập
    )
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
    plot_learning_curve(model, 'Đường cong học tập cho cây quyết định C4.5', X_train, y_train)
    
    # Vẽ ma trận nhầm lẫn
    plot_confusion_matrix(cnf_matrix, classes=['Khỏe mạnh', 'Bệnh tiểu đường'], title='Ma trận nhầm lẫn')
    
    # In kết quả
    print(f'Cây quyết định C4.5 - Độ chính xác tập huấn luyện: {mean_accuracy:.2f} ({std_accuracy:.2f})')
    return model

# Hàm vẽ cây quyết định
def plot_decision_tree(model, feature_names):
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=['Khỏe mạnh', 'Bệnh tiểu đường'],
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    return graph

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

# Đọc dữ liệu
data = pd.read_csv('Data/diabetes.csv')
X = data.drop('Outcome', axis=1)  # Các đặc trưng
y = data['Outcome']  # Biến mục tiêu

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chạy mô hình và trực quan hóa
clf = run_decision_tree(X_train, y_train, X_test, y_test)
save_decision_tree_to_pdf(
    clf,
    feature_names=data.columns[:-1],  # Tên các đặc trưng
    class_names=["Khỏe Mạnh", "Tiểu Đường"],  # Tên các lớp
    output_file="decision_tree_C4.5"  # Tên tệp đầu ra
)

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

model = run_decision_tree(X_train, y_train, X_test, y_test)

# In cây quyết định dưới dạng văn bản
from sklearn.tree import export_text
tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)

# Gọi hàm dự đoán từ dữ liệu người dùng nhập vào
predict_diabetes_from_input(model)
