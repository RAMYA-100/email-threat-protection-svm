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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix 
 
data = { 
    'Email': [ 
        'Free money offer now', 
        'Please find the attached report', 
        'Urgent: Claim your prize', 
        'Meeting at 3 PM', 
        'Win a brand new car!', 
        'Let\'s schedule a call', 
        'Congratulations! You are a winner', 
        'Update your account details' 
    ], 
    'Label': [1, 0, 1, 0, 1, 0, 1, 1] 
} 
 
df = pd.DataFrame(data) 
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Email']) 
y = df['Label'] 
                                                                               
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
svm_model = SVC(kernel='linear', random_state=42) 
svm_model.fit(X_train, y_train) 

y_pred = svm_model.predict(X_test) 

accuracy = accuracy_score(y_test, y_pred) 
precision = precision_score(y_test, y_pred) 
recall = recall_score(y_test, y_pred) 
f1 = f1_score(y_test, y_pred) 

print(f"Accuracy: {accuracy:.2f}") 
print(f"Precision: {precision:.2f}") 
print(f"Recall: {recall:.2f}") 
print(f"F1-Score: {f1:.2f}") 

conf_matrix = confusion_matrix(y_test, y_pred) 
print(f"Confusion Matrix:\n{conf_matrix}")
```
# Output
<img width="402" height="157" alt="image" src="https://github.com/user-attachments/assets/2d6e1a54-93cf-4f8c-9011-2c01a7072d9e" />
