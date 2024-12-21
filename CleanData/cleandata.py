import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import graphviz 
import sys
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve, StratifiedKFold, train_test_split

# 1. Đọc tập dữ liệu từ file csv
print("1. Thu thập dữ liệu...")
data = pd.read_csv('Data/diabetes.csv')
print("   Dữ liệu ban đầu có", data.shape[0], "hàng và", data.shape[1], "cột.")

summary = data.describe()
print(summary)

# 2. Xóa dữ liệu không hợp lệ
print("2. Xóa dữ liệu không hợp lệ")
data.drop_duplicates(inplace=True)
print("   Đã loại bỏ các hàng trùng lặp. Dữ liệu còn", data.shape[0], "hàng.")
data.dropna(inplace=True)
print("   Đã loại bỏ các hàng chứa giá trị thiếu. Dữ liệu còn", data.shape[0], "hàng.")
# Trực quan hóa dữ liệu sau khi loại bỏ hàng trùng lặp và thiếu
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Sự phân bố giá trị thiếu sau khi loại bỏ')
plt.show()



# 4. Lọc dữ liệu
print("4. Lọc dữ liệu...")
important_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = data[important_columns]
print("   Đã chọn các cột quan trọng. Dữ liệu hiện có", data.shape[1], "cột.")

# Trực quan hóa các cột dữ liệu quan trọng: Thay vì sử dụng pairplot, ta sẽ dùng các biểu đồ phân phối cho từng cột

# Hiển thị phân phối cho từng đặc trưng quan trọng
plt.figure(figsize=(15, 12))
for i, col in enumerate(important_columns[:-1]):  # Loại cột 'Outcome' ra vì ta đã có phân loại sẵn
    plt.subplot(3, 3, i + 1)  # Tạo 3x3 lưới biểu đồ
    sns.histplot(data[col], kde=True, bins=20, color='skyblue')
    plt.title(f'Phân phối của {col}')
    plt.tight_layout()
plt.show()

# Sử dụng .isnull() để xác định các giá trị thiếu và .sum() để tính tổng
print("\n Kiểm tra các giá trị thiếu...")
missing_data = data.isnull().sum()
print("   Số lượng giá trị thiếu trong từng cột:\n", missing_data)

total_missing = missing_data.sum()
print(f"\n   Tổng số giá trị thiếu trong toàn bộ dữ liệu: {total_missing}")
# Lưu dữ liệu đã làm sạch
cleaned_file_path = 'Data/cleaned_diabetes.csv'
data.to_csv(cleaned_file_path, index=False)
print(f"\nDữ liệu đã làm sạch và lưu vào: {cleaned_file_path}\n")

# 5. Xử lý dữ liệu thiếu
print("5. Xử lý dữ liệu thiếu")
# Kiểm tra các giá trị bị thiếu hoặc bất thường (0 trong các cột không hợp lý)
missing_values = (data == 0).sum()
missing_values_percentage = (missing_values / len(data)) * 100
# Hiển thị số lượng và phần trăm giá trị bất thường
missing_info = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage (%)': missing_values_percentage
}).sort_values(by='Missing Values', ascending=False)
print(missing_info)
# Trực quan hóa sự phân bố của các cột sau khi xử lý dữ liệu thiếu
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Sự phân bố giá trị thiếu sau khi xử lý')
plt.show()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
imputer = SimpleImputer(missing_values=0, strategy='median')
X_train2 = imputer.fit_transform(X_train)
X_test2 = imputer.transform(X_test)
X_train3 = pd.DataFrame(X_train2)

def plotHistogram(X, y, column_index, title):
    plt.figure(figsize=(8, 6))
    plt.hist(X.iloc[:, column_index], bins=30, color='blue', alpha=0.7, label='Healthy' if y is None else 'Diabetes')
    plt.title(title)
    plt.xlabel(X.columns[column_index])
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

plotHistogram(X_train3, None, 4, 'Insulin vs Diagnosis (Blue = Healthy; Orange = Diabetes)')
plotHistogram(X_train3, None, 3, 'SkinThickness vs Diagnosis (Blue = Healthy; Orange = Diabetes)')

def plotHistogram(values, label, feature, title):
    sns.set_style("whitegrid")
    # Sử dụng histplot thay thế distplot
    plotOne = sns.FacetGrid(values, hue=label, aspect=2)
    plotOne.map(sns.histplot, feature, kde=False)  # Thay distplot bằng histplot
    plotOne.set(xlim=(0, values[feature].max()))
    plotOne.add_legend()
    plotOne.set_axis_labels(feature, 'Proportion')
    plotOne.fig.suptitle(title)
    plt.show()

# Gọi hàm để vẽ biểu đồ
plotHistogram(data, "Outcome", 'Insulin', 'Insulin vs Diagnosis (Xanh = Khỏe Mạnh; Cam = Tiểu Đường)')
plotHistogram(data, "Outcome", 'SkinThickness', 'SkinThickness vs Diagnosis (Xanh = Khỏe Mạnh; Cam = Tiểu Đường)')

# Tính giá trị trung bình của các đặc trưng theo Outcome
mean_values = data.groupby('Outcome')[['Insulin', 'Glucose', 'BloodPressure', 'BMI']].mean()

# Vẽ biểu đồ bar plot cho Insulin
plt.figure(figsize=(8, 6))
sns.barplot(x=mean_values.index, y=mean_values['Insulin'], hue=mean_values.index, palette="muted", legend=False)

plt.title('Trung bình Insulin theo Outcome (0: Khỏe Mạnh, 1: Tiểu Đường)', fontsize=14)
plt.xlabel('Outcome', fontsize=12)
plt.ylabel('Giá trị trung bình Insulin', fontsize=12)
plt.xticks([0, 1], ['Khỏe Mạnh', 'Tiểu đường'], fontsize=10)
plt.show()

# Vẽ các đặc trưng khác
for feature in ['Glucose', 'BloodPressure', 'BMI']:
    plt.figure(figsize=(8, 6))
    sns.barplot(x=mean_values.index, y=mean_values[feature], hue=mean_values.index, palette="muted")
    plt.title(f'Trung bình {feature} theo Outcome (0: Khỏe Mạnh, 1: Tiểu Đường)', fontsize=14)
    plt.xlabel('Outcome', fontsize=12)
    plt.ylabel(f'Giá trị trung bình {feature}', fontsize=12)
    plt.xticks([0, 1], ['Khỏe Mạnh', 'Tiểu đường'], fontsize=10)
    plt.show()

