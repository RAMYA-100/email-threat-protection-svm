# email-threat-protection-svm
A Python-based machine learning system that uses Support Vector Machine (SVM) classification to detect and filter potential cyber threat emails. The project helps protect consumers by improving email security while ensuring smooth communication and uninterrupted workflow through intelligent email analysis and automated threat identification.

# Equipments :
1.Hardware – PCs

2.Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm: 
1.Load the email dataset using pandas.

2.Preprocess email data and remove missing values.

3.Split data into training and testing sets.

4.Predict whether emails are safe or threats.

5.Evaluate and save the trained model.

# program 
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

email_data = {
    'Message': [
        'Free money offer now',
        'Please find the attached report',
        'Urgent: Claim your prize',
        'Meeting at 3 PM',
        'Win a brand new car!',
        'Let\'s schedule a call',
        'Congratulations! You are a winner',
        'Update your account details'
    ],
    'Threat': [1, 0, 1, 0, 1, 0, 1, 1]
}


email_df = pd.DataFrame(email_data)

print("Class Balance:")
print(email_df['Threat'].value_counts())

text_converter = TfidfVectorizer(stop_words='english')
features = text_converter.fit_transform(email_df['Message'])

target = email_df['Threat']


train_x, test_x, train_y, test_y = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42
)


email_classifier = SVC(kernel='linear', random_state=42)
email_classifier.fit(train_x, train_y)


predicted_labels = email_classifier.predict(test_x)


print("\nModel Performance")
print("Accuracy :", round(accuracy_score(test_y, predicted_labels), 2))
print("Precision:", round(precision_score(test_y, predicted_labels), 2))
print("Recall   :", round(recall_score(test_y, predicted_labels), 2))
print("F1 Score :", round(f1_score(test_y, predicted_labels), 2))


matrix = confusion_matrix(test_y, predicted_labels)
print("\nConfusion Matrix:")
print(matrix)

test_messages = email_df.iloc[test_y.index]['Message'].values

print("\nMisclassified Emails:")
found = False

for msg, actual, predicted in zip(test_messages, test_y, predicted_labels):
    if actual != predicted:
        found = True
        print(f"Email: {msg}")
        print(f"Actual: {actual}, Predicted: {predicted}\n")

if not found:
    print("No misclassified emails.")
```
# Output
<img width="590" height="486" alt="Screenshot 2026-06-30 200158" src="https://github.com/user-attachments/assets/27efb051-20e4-49da-9863-9f4da1a2f715" />

