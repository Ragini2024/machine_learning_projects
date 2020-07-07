import numpy as np
import flask as f
import pickle as pkl
import pandas as pd

app = f.Flask(__name__)
model = pkl.load(open("model.pkl", "rb"))

@app.route("/")
def home():
	return f.render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
	A=[]
	for i in f.request.form.values():
		A.append(int(i))
	pred_prof = model.predict(pd.DataFrame([A[0], A[1]]).T)
	return f.render_template("index.html", pred="predicted_profit:%.2f"%pred_prof)

if __name__ == "__main__":
	app.run(debug = True)