# student-marks-prediction
This project uses Linear Regression to predict student exam scores based on the number of study hours. It demonstrates basic machine learning concepts like data preprocessing, model training, and prediction using Python and scikit-learn.
# Student Marks Prediction using Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data: study hours vs marks
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Marks': [35, 40, 50, 60, 65, 70, 80, 85]
}

# Create DataFrame
df = pd.DataFrame(data)

# Split data into input (X) and output (y)
X = df[['Hours']]
y = df['Marks']

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predict marks for given study hours
hours = 5
predicted_marks = model.predict(np.array([[hours]]))

print(f"Predicted marks for {hours} hours of study: {predicted_marks[0]:.2f}")

# Plot data points
plt.scatter(X, y)

# Plot regression line
plt.plot(X, model.predict(X))

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.show()
