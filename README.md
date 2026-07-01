# email-threat-protection-svm

A machine learning system that classifies emails as safe or potential threats using Support Vector Machine (SVM) classification.


## Dataset
UCI SMS Spam Collection — 5,572 real labeled messages (4,825 ham, 747 spam)

## Tech Stack
Python, Scikit-learn, Pandas, TF-IDF Vectorizer, SVM (Linear Kernel)

# Equipments :
1.Hardware – PCs

2.Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm: 
1. Loads real email dataset

2. Converts text to numerical features using TF-IDF

3. Trains a linear SVM classifier
 
4. Evaluates using accuracy, precision, recall, F1 and confusion matrix
 
5. Prints misclassified emails for error analysis

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


# Output
Trained SVM classifier on UCI SMS Spam Collection dataset (5,572 samples). Achieved 97% accuracy and 90% F1-score using TF-IDF vectorization with linear kernel SVM.
| Metric    | Score |
|-----------|-------|
| Accuracy  | 97%   |
| Precision | 97%   |
| Recall    | 84%   |
| F1 Score  | 90%   |

