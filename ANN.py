from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# Example dataset (replace this with your actual dataset loading)
# Features (X): assuming two features for demonstration
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
# Labels (y): assuming binary classification (0 or 1)
y = np.array([0, 0, 1, 1, 1])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naïve Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predict labels
y_pred = nb_classifier.predict(X_test)

# Evaluate performance
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Display results
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy)
