from flask import Flask, render_template,request
import pandas as pd
import joblib
app = Flask(__name__)

model = joblib.load('rf_model.pkl')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data')
def data():
    return render_template('data.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_result = None
    if request.method == 'POST':
        # Get form data
        oldbalanceOrg = float(request.form.get('oldbalanceOrg'))
        newbalanceOrig = float(request.form.get('newbalanceOrig'))
        oldbalanceDest = float(request.form.get('oldbalanceDest'))
        newbalanceDest = float(request.form.get('newbalanceDest'))

        # Perform prediction using the loaded model
        prediction = model.predict([[ oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]])

        # Interpret prediction result
        if prediction[0] == 1:
            prediction_result = "Fraudulent transaction"
        else:
            prediction_result = "Non-fraudulent transaction"

    return render_template('predict.html', prediction_result=prediction_result)

@app.route('/performance')
def performance():
    # Charger les données du fichier CSV
    metrics_df = pd.read_csv('model_metrics.csv')

    # Convertir les données en dictionnaire pour passer à la page HTML
    model_metrics = {}
    for index, row in metrics_df.iterrows():
        model_name = row['Modèle']
        model_metrics[model_name] = {
            'Accuracy': row['Accuracy'],
            'Precision': row['Precision'],
            'Recall': row['Recall'],
            'F1 Score': row['F1 Score']
        }

    return render_template('model_performance.html', model_metrics=model_metrics)



if __name__ == '__main__':
    app.run(debug=True)
