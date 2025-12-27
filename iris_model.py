# Step 1: Import libraries
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Step 2: Load Dataset
iris = load_iris()
X = iris.data
y = iris.target


# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Model
#model = RandomForestClassifier()
model = DecisionTreeClassifier()

# Step 5: Train Model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "iris_model.pkl")

#Load the model (for demonstration)
loaded_model = joblib.load("iris_model.pkl")
print("Loaded Model Prediction:", loaded_model.predict([X_test[0]]))

# Step 6: Predict
y_pred = model.predict(X_test)


# Step 7: Evaluate

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
scores = cross_val_score(model, X, y, cv=5)
print("Cross Validation Scores:", scores)
print("Average Accuracy:", scores.mean())
print(classification_report(y_test, y_pred))