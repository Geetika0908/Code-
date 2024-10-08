import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the CSV file
df = pd.read_csv('C:/Users/sansk/Downloads/extracted_1.csv')

# Assuming the target variable is in a column named 'CLASS'
X = df.drop(columns=['CLASS'])
y = df['CLASS']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Encode the class labels
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Use XGBoost for feature selection
xgb = XGBClassifier(n_estimators=100)  # Set n_estimators
xgb.fit(X_train, y_train_encoded)

# Get feature importances from XGBoost
feature_importances = xgb.feature_importances_

# Select the top k most important features (adjust k as needed)
k = 10  # Number of top features to select
top_k_features_indices = np.argsort(feature_importances)[-k:]

# Subset the data with the selected features
X_train_selected = X_train.iloc[:, top_k_features_indices]
X_test_selected = X_test.iloc[:, top_k_features_indices]

# Train an MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp.fit(X_train_selected, y_train_encoded)

# Make predictions on the test data
y_pred = mlp.predict(X_test_selected)

# Evaluate the classifier's performance
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
