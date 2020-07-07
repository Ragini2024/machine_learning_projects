# Importing essential libraries
from flask import Flask, render_template, request
import pickle as pkl
import flask as f
import numpy as np

# Load the Random Forest CLassifier model
classifier = pkl.load(open('model.pkl', 'rb'))

app = f.Flask(__name__)

@app.route('/')
def home():
	return f.render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        proper = int(request.form['Property_Area'])
        amount = float(request.form['LoanAmount'])
        term = float(request.form['Loan_Amount_Term'])
        credithistory = float(request.form['Credit_History'])
        depend = float(request.form['Dependents'])
        
        data = np.array([[proper, amount, term, credithistory, depend]])
        my_prediction = classifier.predict(data)
        
        return f.render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)