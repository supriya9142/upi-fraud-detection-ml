import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle, os

data = pd.read_csv('dataset/upi_fraud_dataset.csv')

X = data[['amount','oldbalanceOrg','newbalanceOrig']]
y = data['is_fraud']

model = RandomForestClassifier()
model.fit(X, y)

os.makedirs('models', exist_ok=True)
pickle.dump(model, open('models/random_forest_model.pkl','wb'))

print("Model saved")