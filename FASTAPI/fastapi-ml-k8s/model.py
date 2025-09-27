import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,5,4,5])

model = LinearRegression()

joblib.dump(model, 'model.pkl')
print("Model saved to model.pkl")