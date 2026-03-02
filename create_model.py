import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample UPI transaction dataset
data = {
    "amount": [100, 200, 5000, 50, 7000, 300, 10000, 150, 4000, 80],
    "hour": [10, 14, 2, 18, 1, 12, 3, 9, 4, 20],
    "new_device": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "transaction_count": [2, 3, 8, 1, 10, 2, 12, 1, 9, 2],
    "fraud": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
}

df = pd.DataFrame(data)

X = df[["amount", "hour", "new_device", "transaction_count"]]
y = df["fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

print("✅ model.pkl created successfully!")