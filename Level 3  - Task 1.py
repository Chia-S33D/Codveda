# 07 June 2026
# Task 1: Predictive Modelling (Classification)
# Needed: Python, Pandas, Scikit learn, and matplotlib
# Aim: Predict whether a customer will churn using multiple classifiers

# import so many things
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
from sklearn.model_selection import GridSearchCV


# 1. LOAD DATA

train = pd.read_csv(r"C:\Users\User\OneDrive\UCT\Year 3\Codveda\Data Set For Task-2026\Data Set For Task\Churn Prdiction Data\churn-bigml-20.csv")
test = pd.read_csv(r"C:\Users\User\OneDrive\UCT\Year 3\Codveda\Data Set For Task-2026\Data Set For Task\Churn Prdiction Data\churn-bigml-80.csv")

# Dataset overview
print(f"Training set: {train.shape[0]} rows, {train.shape[1]} columns")
print(f"Test set:     {test.shape[0]} rows, {test.shape[1]} columns")
print(f"\nChurn distribution (training):")
print(train['Churn'].value_counts())
print(train.head(3))


# 2. PREPROCESS: HANDLE CATEGORICAL VARIABLES

le = LabelEncoder()
categorical_cols = ['State', 'International plan', 'Voice mail plan']
 
for col in categorical_cols:
    train[col] = le.fit_transform(train[col]) # fit and transform on training data
    test[col]  = le.transform(test[col]) # ONLY transform on test data (use same encoding)

print(train[categorical_cols].head(3))


# 3. SPLIT FEATURES AND TARGET

X_train = train.drop(columns=['Churn']) # X = all columns except Churn (inputs to the model)
y_train = train['Churn']
X_test  = test.drop(columns=['Churn']) # y = Churn column (what we're predicting)
y_test  = test['Churn']


# 4. FEATURE SCALING

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train) # fit AND transform on training data
X_test_sc  = scaler.transform(X_test) # ONLY transform on test data


# 5. TRAIN AND EVALUATE MULTIPLE MODELS

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Random Forest':       RandomForestClassifier(random_state=42)
}
 
results = {}  # store metrics for comparison plot

for name, model in models.items():
    # Train
    model.fit(X_train_sc, y_train)
 
    # Predict
    y_pred = model.predict(X_test_sc)
 
    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
 
    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}

    print(f"  Accuracy:  {acc:.4f}  ← overall % correct")
    print(f"  Precision: {prec:.4f}  ← of predicted churners, how many actually churned")
    print(f"  Recall:    {rec:.4f}  ← of actual churners, how many did we catch")
    print(f"  F1-Score:  {f1:.4f}  ← balance of precision and recall")
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred))


# 6. HYPERPARAMETER TUNING WITH GRID SEARCH

param_grid = {
    'n_estimators': [50, 100, 200],   # number of trees
    'max_depth':    [None, 10, 20]    # how deep each tree can grow
}
 
gs = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,           # 3-fold cross validation
    scoring='f1',   # optimise for F1 score
    verbose=1
)
gs.fit(X_train_sc, y_train)
 
print(f"\nBest parameters: {gs.best_params_}")
print(f"Best CV F1-Score: {gs.best_score_:.4f}")
 
# Evaluate tuned model on test set
best_model = gs.best_estimator_
y_pred_best = best_model.predict(X_test_sc)
print("\nTuned Random Forest - Test Set Report:")
print(classification_report(y_test, y_pred_best))


# 7. VISUALISATIONS

# Plot 1: Model comparison bar chart
metrics_df = pd.DataFrame(results).T
fig, ax = plt.subplots(figsize=(10, 5))
metrics_df.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='black', linewidth=0.5)
ax.set_title('Model Comparison: Accuracy, Precision, Recall, F1-Score')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.1)
ax.set_xticklabels(metrics_df.index, rotation=15, ha='right')
ax.legend(loc='lower right')
ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Plot 2: Feature importances from best Random Forest
importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=True)
 
fig, ax = plt.subplots(figsize=(8, 8))
importances.plot(kind='barh', ax=ax, color='mediumvioletred', edgecolor='black', linewidth=0.4)
ax.set_title('Feature Importances (Tuned Random Forest)')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.show()

