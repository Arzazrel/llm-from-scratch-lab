"""
Adapted from Sebastian Raschka, "Build a Large Language Model (From Scratch)", Apache 2.0. Modified for personal study.

file for establishing a baseline comparison using classification methods that do not employ LLMs
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.dummy import DummyClassifier

# use after downloading the dataset (in gpt_sentiment_analyses_finetune.py)
train_df = pd.read_csv("sent_train.csv")
val_df = pd.read_csv("sent_validation.csv")
test_df = pd.read_csv("sent_test.csv")
train_df.head()

# Text Vectorization (Bag-of-Words)
vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(train_df["text"])    # learn the vocabulary from the training set and transform the texts into vectors
# apply the same vocabulary to the validation and test sets without recalculating it
X_val = vectorizer.transform(val_df["text"])            
X_test = vectorizer.transform(test_df["text"])
y_train, y_val, y_test = train_df["label"], val_df["label"], test_df["label"]

# Calculate predictions and metrics (accuracy and balanced accuracy) across all sets and print the results as percentages
def eval(model, X_train, y_train, X_val, y_val, X_test, y_test):
    # making predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    # Calculating accuracy and balanced accuracy
    accuracy_train = accuracy_score(y_train, y_pred_train)
    balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)
    
    accuracy_val = accuracy_score(y_val, y_pred_val)
    balanced_accuracy_val = balanced_accuracy_score(y_val, y_pred_val)

    accuracy_test = accuracy_score(y_test, y_pred_test)
    balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)
    
    # Printing the results
    print(f"Training Accuracy: {accuracy_train*100:.2f}%")
    print(f"Validation Accuracy: {accuracy_val*100:.2f}%")
    print(f"Test Accuracy: {accuracy_test*100:.2f}%")



# Create a dummy classifier with the strategy to predict the most frequent class. 
# It serves as a baseline for comparison: any intelligent model must perform significantly better than this.
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
print("Dummy algorithm: ")
eval(dummy_clf, X_train, y_train, X_val, y_val, X_test, y_test) # evaluate dummy model

# linear regression algorithm
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("\nLinear regression algorithm: ")
eval(model, X_train, y_train, X_val, y_val, X_test, y_test)     # evaluate regression model
