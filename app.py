from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import joblib
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load models
logistic_regression_balanced_no_fs_model = joblib.load('logistic_regression_balanced_no_fs_model.pkl')
logistic_regression_balanced_fs_model = joblib.load('logistic_regression_balanced_fs_model.pkl')
random_forest_balanced_no_fs_model = joblib.load('random_forest_balanced_no_fs_model.pkl')
random_forest_balanced_fs_model = joblib.load('random_forest_balanced_fs_model_rs.pkl')

# Load dataset
data = pd.read_csv("C:\\Users\\LENOVO\\Downloads\\Online_Payments_Fraud_Detection.csv")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data')
def data_page():
    return render_template('data.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        model = request.form['model']
        feature_selection = request.form['feature_selection']
        if model == 'logistic':
            if feature_selection == 'with':
                return render_template('logistic_with_fs.html')
            else:
                return render_template('logistic_without_fs.html')
        elif model == 'random_forest':
            if feature_selection == 'with':
                return render_template('random_forest_with_fs.html')
            else:
                return render_template('random_forest_without_fs.html')
    return render_template('predict.html')

selected_features = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

@app.route('/logistic_with_fs')
def logistic_with_fs():
    prediction_result = ""
    return render_template('logistic_with_fs.html', prediction_result=prediction_result)

@app.route('/predict_logistic_with_fs', methods=['POST'])
def predict_logistic_with_fs():
    # Get user input from form
    oldbalanceOrg = float(request.form['oldbalanceOrg'])
    newbalanceOrig = float(request.form['newbalanceOrig'])
    oldbalanceDest = float(request.form['oldbalanceDest'])
    newbalanceDest = float(request.form['newbalanceDest'])

    # Prepare input data for prediction
    input_data = np.array([[oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]])

    # Make prediction
    prediction = logistic_regression_balanced_fs_model.predict(input_data)

    # Convert prediction to human-readable label
    prediction_label = 'Fraud' if prediction == 1 else 'Non-Fraud'

    # Construct prediction result message
    prediction_result = f"Prediction: {prediction_label}"

    # Generate a plot to show the transaction position relative to dataset
    fig, ax = plt.subplots()

    # Scatter plot of dataset sample
    sample_data = data.sample(n=1000)  # Sample 1000 points from your dataset
    ax.scatter(sample_data['oldbalanceOrg'], sample_data['newbalanceOrig'], color='blue', alpha=0.5, label='Dataset Sample')

    # Scatter plot of predicted transaction
    ax.scatter(oldbalanceOrg, newbalanceOrig, color='red', label='Transaction')

    ax.set_xlabel('Old Balance Orig')
    ax.set_ylabel('New Balance Orig')
    ax.set_title('Transaction Position vs Dataset Sample')
    ax.legend()

    # Save plot to a bytes object
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)

    # Encode plot to base64
    plot_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')

    # Render the template with prediction result and graph
    return render_template('logistic_with_fs.html', prediction_result=prediction_result, plot_base64=plot_base64)

@app.route('/logistic_without_fs')
def logistic_without_fs():
    prediction_result = ""
    return render_template('logistic_without_fs.html', prediction_result=prediction_result)

@app.route('/predict_logistic_without_fs', methods=['POST'])
def predict_logistic_without_fs():
    oldbalanceOrg = float(request.form['oldbalanceOrg'])
    newbalanceOrig = float(request.form['newbalanceOrig'])
    oldbalanceDest = float(request.form['oldbalanceDest'])
    amount = float(request.form['amount'])
    newbalanceDest = float(request.form['newbalanceDest'])

    input_data = np.array([[oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, amount]])
    prediction = logistic_regression_balanced_no_fs_model.predict(input_data)
    prediction_label = 'Fraud' if prediction == 1 else 'Non-Fraud'
    prediction_result = f"Prediction: {prediction_label}"

    return render_template('logistic_without_fs.html', prediction_result=prediction_result)

@app.route('/random_forest_with_fs')
def random_forest_with_fs():
    prediction_result = ""
    return render_template('random_forest_with_fs.html', prediction_result=prediction_result)

@app.route('/predict_random_forest_with_fs', methods=['POST'])
def predict_random_forest_with_fs():
    oldbalanceOrg = float(request.form['oldbalanceOrg'])
    newbalanceOrig = float(request.form['newbalanceOrig'])
    oldbalanceDest = float(request.form['oldbalanceDest'])
    newbalanceDest = float(request.form['newbalanceDest'])

    input_data = np.array([[oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]])
    prediction = random_forest_balanced_fs_model.predict(input_data)
    prediction_label = 'Fraud' if prediction == 1 else 'Non-Fraud'
    prediction_result = f"Prediction: {prediction_label}"

    return render_template('random_forest_with_fs.html', prediction_result=prediction_result)

@app.route('/random_forest_without_fs')
def random_forest_without_fs():
    prediction_result = ""
    return render_template('random_forest_without_fs.html', prediction_result=prediction_result)

@app.route('/predict_random_forest_without_fs', methods=['POST'])
def predict_random_forest_without_fs():
    oldbalanceOrg = float(request.form['oldbalanceOrg'])
    newbalanceOrig = float(request.form['newbalanceOrig'])
    oldbalanceDest = float(request.form['oldbalanceDest'])
    amount = float(request.form['amount'])
    newbalanceDest = float(request.form['newbalanceDest'])

    input_data = np.array([[oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, amount]])
    prediction = random_forest_balanced_no_fs_model.predict(input_data)
    prediction_label = 'Fraud' if prediction == 1 else 'Non-Fraud'
    prediction_result = f"Prediction: {prediction_label}"

    return render_template('random_forest_without_fs.html', prediction_result=prediction_result)

@app.route('/performance')
def performance():
    metrics_df = pd.read_csv('model_metrics.csv')

    model_metrics = {}
    for index, row in metrics_df.iterrows():
        model_name = row['Model']
        model_metrics[model_name] = {
            'Accuracy': row['Accuracy'],
            'Precision': row['Precision'],
            'Recall': row['Recall'],
            'F1 Score': row['F1 Score']
        }

    return render_template('model_performance.html', model_metrics=model_metrics)

if __name__ == '__main__':
    app.run(debug=True)
