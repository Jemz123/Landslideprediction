import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = r"C:\Users\Administrator\Desktop\pythonprojects\landslide.csv" # Replace with the correct file path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Selecting relevant features for prediction
selected_columns = [
    'landslide_trigger', 'landslide_size', 'landslide_setting', 'fatality_count', 'injury_count'
]
data = data[selected_columns + ['landslide_category']]  # Adding target column

# Data preprocessing
# Convert all entries in `landslide_category` to strings for uniformity
data['landslide_category'] = data['landslide_category'].astype(str)

# Encoding categorical variables
data = pd.get_dummies(data, columns=['landslide_trigger', 'landslide_size', 'landslide_setting'], drop_first=True)

# Handling missing values
data.fillna(0, inplace=True)

# Splitting data into features and target
X = data.drop(columns=['landslide_category'])  # Features
y = data['landslide_category']  # Target

# Encoding the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification Report with correct label mapping
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_[np.unique(y_test)]))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
