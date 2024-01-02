import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'diabetes.csv'
data = pd.read_csv(file_path)

# Remove 'Outcome' column and set 'DiabetesPedigreeFunction' as the target variable
X = data.drop(['Outcome', 'DiabetesPedigreeFunction'], axis=1)
y = data['DiabetesPedigreeFunction']

# Splitting dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize SVM with different kernel settings
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svm_reg_results = {}

# Apply SVM regressor using several kernel settings
for kernel in kernels:
    svm_reg = SVR(kernel=kernel)
    svm_reg.fit(X_train, y_train)
    y_pred = svm_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    svm_reg_results[kernel] = mse

# Print MSE results for each kernel
for kernel, mse in svm_reg_results.items():
    print(f"SVM Regressor MSE with {kernel} kernel: {mse:.4f}")

# Setting up the hyperparameter grids for each kernel
param_grid = {
    'linear': {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 1]},
    'poly': {'C': [0.1, 1, 10, 100], 'degree': [2, 3, 4], 'epsilon': [0.01, 0.1, 1]},
    'rbf': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'epsilon': [0.01, 0.1, 1]},
    'sigmoid': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'epsilon': [0.01, 0.1, 1]}
}

best_params_results = {}
best_scores_results = {}

# Perform cross-validation and parameter tuning for each kernel
for kernel in kernels:
    grid_search = GridSearchCV(SVR(kernel=kernel), param_grid[kernel], cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_params_results[kernel] = grid_search.best_params_
    best_scores_results[kernel] = -grid_search.best_score_  # Negative MSE to positive

# Print the best parameters and scores for each kernel
for kernel in best_params_results:
    print(f"Best parameters for SVM with {kernel} kernel: {best_params_results[kernel]}")
    print(f"Best cross-validation MSE for SVM with {kernel} kernel: {best_scores_results[kernel]:.4f}")
