Anemia Detection 
===========================

A simple Flask web application that predicts whether a patient is likely to suffer from anemia based on their gender, hemoglobin level, MCH (Mean Corpuscular Hemoglobin), MCHC (Mean Corpuscular Hemoglobin Concentration), and MCV (Mean Corpuscular Volume).

Table of Contents
-----------------

* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
* [Model](#model)
* [License](#license)

Requirements
------------

* Python 3.x
* Flask
* scikit-learn
* pandas
* joblib

Installation
------------

1. Clone the repository:
```bash
git clone https://github.com/your-username/anemia-detection-flask-app.git
```
2. Change to the project directory:
```bash
cd anemia-detection-flask-app
```
3. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
```
4. Activate the virtual environment:
```bash
source venv/bin/activate (Unix)
venv\Scripts\activate (Windows)
```
5. Install the required packages:
```bash
pip install -r requirements.txt
```

Usage
-----

1. Run the Flask application:
```bash
python app.py
```
2. Open your web browser and navigate to `http://127.0.0.1:5000/` to access the Anemia Detection web application.
3. Enter the patient's gender (0 - male, 1 - female), hemoglobin level, MCH, MCHC, and MCV values in the form, and click "Detect Anemia" to get the prediction.

Model
-----

The machine learning model used in this application is a logistic regression model trained on a dataset containing information about patients, including their gender, hemoglobin level, MCH, MCHC, and MCV values. The model predicts whether a patient is likely to suffer from anemia based on these features.

The dataset used to train the model is not included in this repository but can be found in [this example](https://example.com/anemia-dataset).

License
-------

This project is licensed under the [MIT License](LICENSE).