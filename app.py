from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('anemia_model.pkl')
scaler = joblib.load('anemia_scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        gender = int(request.form['gender'])
        hemoglobin = float(request.form['hemoglobin'])
        mch = float(request.form['mch'])
        mchc = float(request.form['mchc'])
        mcv = float(request.form['mcv'])

        # Prepare input data for the model
        input_data = np.array([gender, hemoglobin, mch, mchc, mcv]).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Display the result
        if prediction[0] == 0:
            result = "Not Anemic"
        else:
            result = "Anemic"

        return render_template('index.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
