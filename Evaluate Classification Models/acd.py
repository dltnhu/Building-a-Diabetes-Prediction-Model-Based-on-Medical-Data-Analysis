import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier

# Đọc dữ liệu và chia thành features (X) và labels (y)
data = pd.read_csv('g:/DA/DULIEU/DULIEU/cleaned_diabetes.csv')
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
clf1 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=12, random_state=42)
clf1.fit(X_train2, y_train)
print('Accuracy of DecisionTreeClassifier: {:.2f}'.format(clf1.score(X_test2, y_test)))
columns = X.columns
coefficients = clf1.feature_importances_.reshape(X.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns=['Variable']), pd.DataFrame(absCoefficients, columns=['absCoefficient'])), axis=1).sort_values(by='absCoefficient', ascending=False)
print('DecisionTreeClassifier - Feature Importance:')
print('\n', fullList, '\n')

# Random Forest Classifier
clf2 = RandomForestClassifier(max_depth=3, min_samples_leaf=12, random_state=42)
clf2.fit(X_train2, y_train)
print('Accuracy of RandomForestClassifier: {:.2f}'.format(clf2.score(X_test2, y_test)))
coefficients = clf2.feature_importances_.reshape(X.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns=['Variable']), pd.DataFrame(absCoefficients, columns=['absCoefficient'])), axis=1).sort_values(by='absCoefficient', ascending=False)
print('RandomForestClassifier - Feature Importance:')
print('\n', fullList, '\n')
