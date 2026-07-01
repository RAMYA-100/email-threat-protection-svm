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

email_df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
email_df.columns = ['Threat', 'Message']

print("Class Balance:")
print(email_df['Threat'].value_counts())

text_converter = TfidfVectorizer(stop_words='english')
features = text_converter.fit_transform(email_df['Message'])
target = email_df['Threat']

train_x, test_x, train_y, test_y = train_test_split(
    features, target, test_size=0.2, random_state=42
)

email_classifier = SVC(kernel='linear', random_state=42)
email_classifier.fit(train_x, train_y)

predicted_labels = email_classifier.predict(test_x)

print("\nModel Performance")
print("Accuracy :", round(accuracy_score(test_y, predicted_labels), 2))
print("Precision:", round(precision_score(test_y, predicted_labels, pos_label='spam'), 2))
print("Recall   :", round(recall_score(test_y, predicted_labels, pos_label='spam'), 2))
print("F1 Score :", round(f1_score(test_y, predicted_labels, pos_label='spam'), 2))

matrix = confusion_matrix(test_y, predicted_labels, labels=['ham', 'spam'])
print("\nConfusion Matrix (rows=Actual, cols=Predicted):")
print(f"              ham   spam")
print(f"  ham     {matrix[0]}")
print(f"  spam    {matrix[1]}")

# Bug fix: use .loc not .iloc
test_messages = email_df.loc[test_y.index]['Message'].values

print("\nMisclassified Emails (first 5):")
found = False
count = 0

for msg, actual, predicted in zip(test_messages, test_y, predicted_labels):
    if actual != predicted:
        found = True
        count += 1
        print(f"Email    : {msg}")
        print(f"Actual   : {actual}  |  Predicted: {predicted}\n")
        if count == 5:
            break

if not found:
    print("No misclassified emails.")
```
# Output
<img width="1653" height="717" alt="image" src="https://github.com/user-attachments/assets/1a38fe5f-bd8f-43e7-8ef0-d553f4a3f03b" />


