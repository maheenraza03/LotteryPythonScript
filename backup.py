import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
# Replace 'lottery_dataset.csv' with the actual path to your downloaded dataset
data = pd.read_csv('649.csv')

# Example preprocessing: Assuming the dataset has columns for each number
# Extract features and labels (target)
# Note: The dataset structure will determine the exact steps
# Assuming 'Number1', 'Number2', ..., 'NumberN' are columns for the numbers

# For demonstration, let's create a 'target' column as the sum of the numbers
data['target'] = data.iloc[:, :6].sum(axis=1)  # Assuming 6 numbers per draw

# Split the data into features and target
X = data.iloc[:, :6]  # All number columns
y = data['target']  # Target: Sum of numbers (example)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict the next set of numbers (example: using the last row of X_test)
next_prediction = model.predict([X_test.iloc[-1]])
print(f"Next predicted number (sum): {next_prediction[0]}")
