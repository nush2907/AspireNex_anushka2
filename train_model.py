import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading dataset...")
df = pd.read_csv('creditcard.csv')

print("Splitting data...")
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Scaling data...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Saving model and scaler...")
joblib.dump(model, 'random_forest_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler saved successfully.")
