import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Step 1: Load the dataset
# Assuming you have the Titanic dataset in the same directory
train_df = pd.read_csv('titanic_train.csv')

# Step 2: Data Exploration (check the first few rows and data info)
print(train_df.head())
print(train_df.info())

# Step 3: Handle missing values
# Filling missing Age values with the mean
train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
# Filling missing Embarked values with the mode (most frequent value)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Step 4: Convert categorical features to numeric values using LabelEncoder
label_encoder = LabelEncoder()
train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'])
train_df['Embarked'] = label_encoder.fit_transform(train_df['Embarked'])

# Step 5: Select the features (independent variables) and target (dependent variable)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_df[features]  # Features
y = train_df['Survived']  # Target variable

# Step 6: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed for convergence
model.fit(X_train, y_train)

# Step 8: Predict the target values on the test set
y_pred = model.predict(X_test)

# Step 9: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Step 10: Print classification report (precision, recall, f1-score, etc.)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 11: Visualize the confusion matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()
