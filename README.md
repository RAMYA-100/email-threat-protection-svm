

# Email Threat Protection using SVM

A machine learning system that classifies emails as safe or potential threats using Support Vector Machine (SVM) classification.

## Dataset
UCI SMS Spam Collection — 5,572 real labeled messages (4,825 ham, 747 spam)

## Tech Stack
Python, Scikit-learn, Pandas, TF-IDF Vectorizer, SVM (Linear Kernel)

## Results
| Metric    | Score |
|-----------|-------|
| Accuracy  | 97%   |
| Precision | 97%   |
| Recall    | 84%   |
| F1 Score  | 90%   |

## How it works
1. Loads real email dataset
2. Converts text to numerical features using TF-IDF
3. Trains a linear SVM classifier
4. Evaluates using accuracy, precision, recall, F1 and confusion matrix
5. Prints misclassified emails for error analysis

## How to run
```python
pip install pandas scikit-learn
python email_threat_svm.py
```
