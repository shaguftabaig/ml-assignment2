import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'diabetes.csv'
data = pd.read_csv(file_path)

# Remove the 'DiabetesPedigreeFunction' column
data = data.drop('DiabetesPedigreeFunction', axis=1)

# Define features (X) and target variable (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Splitting dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVM with different kernel settings
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svm_results = {}

# Apply SVM classifier using several kernel settings
for kernel in kernels:
    svm_clf = SVC(kernel=kernel)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    svm_results[kernel] = accuracy

for kernel, accuracy in svm_results.items():
    print(f"SVM Accuracy with {kernel} kernel: {accuracy * 100:.2f}%")

# Setting up the hyperparameter grid for the RBF kernel
param_grid_svm_rbf = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale']
}

# Cross-validation and hyperparameter tuning for SVM with the RBF kernel
grid_search_svm_rbf = GridSearchCV(SVC(kernel='rbf'), param_grid_svm_rbf, cv=5)
grid_search_svm_rbf.fit(X_train, y_train)
best_params_svm_rbf = grid_search_svm_rbf.best_params_
best_score_svm_rbf = grid_search_svm_rbf.best_score_

print(f"Best hyperparameters for SVM with RBF kernel: {best_params_svm_rbf}")

print(f"Best cross-validation score for SVM with RBF kernel: {best_score_svm_rbf * 100:.2f}%")
