from scipy import stats
from sklearn.metrics import r2_score
import pandas as pd
import pickle as pkl

A=pd.read_csv("50_Startups.csv")
X=A[["RND", "MKT"]]
Y=A[["PROFIT"]]

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X,Y)

pkl.dump(lm, open("model.pkl", "wb"))
