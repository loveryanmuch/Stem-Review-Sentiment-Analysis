import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("C:\\Users\\17789\\LHL\\Stem-Review-Sentiment-Analysis\\Data\\Filtered_data.csv")

X = df['Review_text'] 
y = df['review_score'] 

predictions = model.predict(X_test)

# Create and display a confusion matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Calculate and print evaluation metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='binary')  # Change to 'macro' if multi-class
recall = recall_score(y_test, predictions, average='binary')  # Change to 'macro' if multi-class
f1 = f1_score(y_test, predictions, average='binary')  # Change to 'macro' if multi-class

print(f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}")

# Hyperparameter tuning
parameters = {'learning_rate': [0.01, 0.001], 'batch_size': [16, 32], 'epochs': [3, 5]}
grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameters found by GridSearchCV
best_parameters = grid_search.best_params_
print(f"Best Parameters: {best_parameters}")
