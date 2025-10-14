# ----------------------------------------------------------------------------------
# Building a Basic Sepsis Prediction Model
# ----------------------------------------------------------------------------------

# 1. IMPORTS: Load toolkits
import pandas as pd              # Used for handling data tables (like our CSV file).
from sklearn.model_selection import train_test_split # Tool to split data into training/testing sets.
from sklearn.linear_model import LogisticRegression    # The simple AI model we will use first.
from sklearn.metrics import accuracy_score, classification_report # Tools to measure how good the model is.

print("Toolkits loaded successfully!")

# 2. DATA LOADING: Load simulated patient data using the 'pandas' library.
try:
    df = pd.read_csv('mock_sepsis_data.csv')
    print("Data loaded successfully! Head of data (first 5 rows):")
    # This line prints the top 5 rows so we can inspect the data structure.
    print(df.head())
except FileNotFoundError:
    print("Error: The file 'mock_sepsis_data.csv' was not found. Make sure it's in the same folder!")
    exit()

# 3. DATA PREPARATION: Define inputs (features) and output (target).

# X (Features) contains the vital signs used to predict sepsis onset.
#  Drop Patient_ID (not useful) and Sepsis_Onset (the answer).
X = df.drop(['Patient_ID', 'Sepsis_Onset'], axis=1)

# y (Target) contains the answer the model must learn to predict (0=No Sepsis, 1=Sepsis).
y = df['Sepsis_Onset']

print("\nFeatures (X) ready for training. First 5 patients' vitals:")
print(X.head())
print("\nTarget (y) ready for training. First 5 patients' sepsis outcome:")
print(y.head())

# 4. SPLIT DATA: Divide the data into a Training set (for learning) and a Test set (for evaluation).
# Using 80% for training and 20% for testing (test_size=0.2).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42 # Ensures the split is the same every time we run the code.
)

print(f"\nData split complete. Training size: {len(X_train)} samples, Testing size: {len(X_test)} samples.")


# 5. TRAIN THE MODEL: Initialize and train the Logistic Regression model.
model = LogisticRegression(solver='liblinear', random_state=42)

# This line is where the AI 'learns' the relationship between vitals (X_train) and outcome (y_train).
model.fit(X_train, y_train)

print("AI Model (Logistic Regression) training complete!")

# 6. EVALUATE THE MODEL: Check how well the model performed on the test data.

# Use the trained model to make predictions on the test set vitals (X_test).
y_pred = model.predict(X_test)

# Calculate the model's accuracy (how many predictions were correct).
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Performance Results ---")
print(f"Accuracy Score: {accuracy * 100:.2f}%")
print("\nClassification Report (Detailed Performance):")
# The classification report shows precision, recall, and f1-score for Sepsis (1) and No Sepsis (0).
print(classification_report(y_test, y_pred))
print("-----------------------------------")
