import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import os
import time

iter = 2000
test = 0.3
randomly = 42

print("Iteration Count: ",iter)
print("Test / Traning set rate: ",test)
print("Randomness in Splitting the Data Set: ",randomly)
print("\nCalculating Accuracy...\n")

# Load dataset
data = pd.read_csv("diabetes_prediction_dataset.csv")

# Fill in missing values
data.ffill(inplace=True)  # fillna yerine ffill kullanıldı

# Process categorical variables
data = pd.get_dummies(data, columns=['gender', 'smoking_history'])

# Hedef değişkeni ayır
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Allocate target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=randomly)

# Define and train the model
model = LogisticRegression(max_iter=iter)
model.fit(X_train, y_train)

# Evaluate the performance of the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("-------------------------------")
print("-------------------------------")
print("Accuracy:", round(accuracy,4))
print("-------------------------------")
time.sleep(0.5)
os.system("cls")
print("-------------------------------")
print("-------------------------------")
print("Model Accuracy:", round(accuracy,4))
print("-------------------------------")

# Randomly generate a new person's data
new_data = {
    'age': np.random.randint(20, 90),
    'hypertension': np.random.choice([0, 1]),
    'heart_disease': np.random.choice([0, 1]),
    'bmi': np.random.uniform(15, 45),
    'HbA1c_level': np.random.uniform(4, 10),
    'blood_glucose_level': np.random.randint(50, 250),
    'gender_Female': np.random.choice([0, 1]),
    'gender_Male': np.random.choice([0, 1]),
    'smoking_history_No Info': np.random.choice([0, 1]),
    'smoking_history_never': np.random.choice([0, 1]),
}

# Convert new data to DataFrame
new_data_df = pd.DataFrame([new_data])

# Access feature names used in the model
feature_names = X.columns

# Add feature names used in the model to the new data set
new_data_df = new_data_df.reindex(columns=feature_names, fill_value=0)

# Make a guess
prediction = model.predict(new_data_df)

# Predict the probability of having diabetes
probabilities = model.predict_proba(new_data_df)

# Print the probability of diabetes on the screen
diabetes_probability = probabilities[0][1]  # Probability corresponding to class 1

print("-------------------------------")
# Print each feature of a randomly generated person on the screen, one under the other
print("Randomly Generated Person Data:")
for key, value in new_data.items():
    print(f"{key}: {value}")
print("-------------------------------")
print("The probability that this person has diabetes is %:", 100*diabetes_probability)