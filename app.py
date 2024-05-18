from flask import Flask, render_template
from sklearn.model_selection import train_test_split
#from .model import *


app = Flask(__name__)

# Charger le modèle sauvegardé
#model = joblib.load('rf_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')
#
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     data = request.get_json()
# #     df = pd.DataFrame(data, index=[0])
# #     prediction = model.predict(df)
# #     result = {'isFraud': bool(prediction[0])}
# #     return jsonify(result)
# @app.route('/data')
# def data():
#     # Charger les données
#     data = load_data("C:\\Users\\LENOVO\\Downloads\\Online_Payments_Fraud_Detection.csv")
#     # Visualiser les données
#     visualize_transaction_amounts(data)
#     # Équilibrer les données
#     balanced_data = balance_data(data)
#     # Visualiser la distribution des données avant et après équilibrage
#     visualize_data_distribution_before_after(data, balanced_data)
#     # Corrélation entre les caractéristiques
#     correlation_matrix(balanced_data)
#
#     # Retourner une réponse
#     return render_template('index.html')
#
#
# @app.route('/model_performance')
# def model_performance():
#     # Charger les données
#     data = load_data("C:\\Users\\LENOVO\\Downloads\\Online_Payments_Fraud_Detection.csv")
#     # Sélectionner les caractéristiques
#     selected_features = select_features(data)
#     # Séparer les données en ensembles d'entraînement et de test
#     X_balanced = selected_features.drop('isFraud', axis=1)
#     y_balanced = selected_features['isFraud']
#     X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.25, random_state=1,
#                                                         stratify=y_balanced)
#     # Entraîner le modèle de régression logistique
#     log_reg_model = train_logistic_regression(X_train, y_train)
#     # Entraîner le modèle de forêt aléatoire
#     rf_model = train_random_forest(X_train, y_train)
#     # Effectuer des prédictions
#     log_reg_predictions = train_and_predict(log_reg_model, X_train, y_train, X_test)
#     rf_predictions = train_and_predict(rf_model, X_train, y_train, X_test)
#     # Évaluer les performances des modèles
#     log_reg_metrics = evaluate_model(log_reg_predictions, y_test)
#     rf_metrics = evaluate_model(rf_predictions, y_test)
#     # Recherche aléatoire pour les hyperparamètres
#     best_params = random_search(X_train, y_train)
#     # Entraîner le modèle de forêt aléatoire avec les meilleurs hyperparamètres
#     rf_hyper_model = RandomForestClassifier(**best_params)
#     rf_hyper_model.fit(X_train, y_train)
#     rf_hyper_predictions = rf_hyper_model.predict(X_test)
#     rf_hyper_metrics = evaluate_model_hyper(rf_hyper_predictions, y_test)
#
#     # Retourner une réponse
#     return render_template('index.html', log_reg_metrics=log_reg_metrics, rf_metrics=rf_metrics,
#                            rf_hyper_metrics=rf_hyper_metrics)
#

if __name__ == '__main__':
    app.run(debug=True)
