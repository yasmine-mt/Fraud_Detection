import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample


def load_data(file_path):
    return pd.read_csv(file_path)

def explore_data(data):
    print(data.head())
    print(data.info())
    print(data['isFraud'].value_counts())

def visualize_transaction_amounts(data):
    plt.figure(figsize=(10, 5))
    sns.histplot(data[data['isFraud'] == 1]['amount'], color='red', label='Fraudulent', kde=True)
    sns.histplot(data[data['isFraud'] == 0]['amount'], color='blue', label='Non-fraudulent', kde=True)
    plt.title('Distribution of Transaction Amounts')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def balance_data(data):
    fraudulent_transactions = data[data['isFraud'] == 1]
    non_fraudulent_transactions = data[data['isFraud'] == 0]
    total_transactions_needed = len(non_fraudulent_transactions) * 10 // 9
    fraudulent_count_needed = total_transactions_needed - len(non_fraudulent_transactions)

    fraudulent_oversampled = resample(fraudulent_transactions,
                                      replace=True,
                                      n_samples=fraudulent_count_needed,
                                      random_state=42)

    balanced_data = pd.concat([non_fraudulent_transactions, fraudulent_oversampled])
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_data

def check_missing_values(data):
    return data.isnull().sum()

def visualize_data_distribution_before_after(data, balanced_data):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x='isFraud', data=data)
    plt.title('Data Distribution (Before Balancing)')
    plt.subplot(1, 2, 2)
    sns.countplot(x='isFraud', data=balanced_data)
    plt.title('Data Distribution (After Balancing)')
    plt.tight_layout()
    plt.show()

def correlation_matrix(data):
    numeric_data = data.select_dtypes(include=['number'])
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def select_features(data):
    return data[['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']]

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(solver='lbfgs', max_iter=500)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(predictions, y_test):
    metrics = {
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions, average='weighted'),
        'Recall': recall_score(y_test, predictions, average='weighted'),
        'F1 Score': f1_score(y_test, predictions, average='weighted')
    }
    return metrics

def train_and_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)

def random_search(X_train, y_train):
    param_grid = {
        'n_estimators': [100],
        'max_features': ['sqrt'],
        'max_depth': [6],
        'criterion': ['gini'],
        'max_leaf_nodes': [6],
        'min_samples_leaf': [2],
        'min_samples_split': [5],
    }

    random_search_cv = RandomizedSearchCV(estimator=RandomForestClassifier(),
                                         param_distributions=param_grid,
                                         n_iter=30, cv=3, verbose=4,
                                         random_state=42, n_jobs=-1)
    random_search_cv.fit(X_train, y_train)
    return random_search_cv.best_params_

def evaluate_model_hyper(predictions, y_test):
    metrics = {
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions, average='weighted'),
        'Recall': recall_score(y_test, predictions, average='weighted'),
        'F1 Score': f1_score(y_test, predictions, average='weighted')
    }
    return metrics
