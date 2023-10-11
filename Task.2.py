# Import necessary libraries
import numpy as np
import pandas as pd
import warnings  # Import warnings module
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load your custom dataset into a DataFrame
# Replace 'C:\\Users\\amanv\\Downloads\\Iris.csv' with the correct file path
data = pd.read_csv('C:\\Users\\amanv\\Downloads\\Iris.csv')

# Extract features (X) and target labels (y)
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['Species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (optional but recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train a Random Forest Classifier
random_forest_classifier = RandomForestClassifier(random_state=42)
random_forest_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = random_forest_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Input new measurements from the user
sepal_length = float(input("Enter sepal length (cm): "))
sepal_width = float(input("Enter sepal width (cm): "))
petal_length = float(input("Enter petal length (cm): "))
petal_width = float(input("Enter petal width (cm): "))

# Suppress warnings
warnings.filterwarnings("ignore")

# Create a new scaler for the new data point
new_data_point = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])

# Make a prediction using the trained classifier
predicted_species = random_forest_classifier.predict(new_data_point)

print(f"The predicted species is: {predicted_species[0]}")

# Restore warnings
warnings.resetwarnings()

