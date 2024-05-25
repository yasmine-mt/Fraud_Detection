from flask import Flask, render_template
import pandas as pd
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route('/performance')
def performance():
    # Charger les données du fichier CSV
    metrics_df = pd.read_csv('model_metrics_hyper.csv')

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
