"""
Diabetes Prediction Model
This script implements a machine learning model to predict diabetes based on various health metrics from the NHANES dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Step 1: Data Loading and Preparation
print("Loading and preparing data...")

# Load the XPT files
diabetes_ques = pd.read_sas('DIQ_L.xpt', format='xport')
fasting_glucose = pd.read_sas('GLU_L.xpt', format='xport')
insulin = pd.read_sas('INS_L.xpt', format='xport')
tot_cholesterol = pd.read_sas('TCHOL_L.xpt', format='xport')
weight_hist = pd.read_sas('WHQ_L.xpt', format='xport')
demographics = pd.read_sas('DEMO_L.xpt', format='xport')

# Select relevant columns
demographics = demographics[['SEQN', 'RIAGENDR', 'RIDAGEYR']]
diabetes_ques = diabetes_ques[['SEQN', 'DIQ160']]
fasting_glucose = fasting_glucose[['SEQN', 'LBXGLU']]
insulin = insulin[['SEQN', 'LBDINSI']]
tot_cholesterol = tot_cholesterol[['SEQN', 'LBXTC']]
weight_hist = weight_hist[['SEQN', 'WHD010', 'WHD020']]

# Calculate BMI
weight_hist['BMI'] = (weight_hist['WHD020'] / (weight_hist['WHD010'] * weight_hist['WHD010'])) * 703
weight_hist['is_overweight'] = weight_hist['BMI'].apply(lambda x: 1 if x >= 30 else 0)

# Merge all relevant features
print("Merging features...")
final_data = pd.merge(demographics, fasting_glucose, on='SEQN', how='inner')
final_data = pd.merge(final_data, insulin, on='SEQN', how='inner')
final_data = pd.merge(final_data, tot_cholesterol, on='SEQN', how='inner')
final_data = pd.merge(final_data, weight_hist[['SEQN', 'BMI', 'is_overweight']], on='SEQN', how='inner')

# Calculate HOMA-IR
final_data['HOMA_IR'] = (final_data['LBDINSI'] * final_data['LBXGLU']) / 405

# Create diabetes measurement column based on fasting glucose levels
# A fasting glucose level of 126 mg/dL or higher indicates diabetes
final_data['diabetes_meas'] = final_data['LBXGLU'].apply(lambda x: 1 if x >= 126 else 0)

# Step 2: Feature Selection and Target Definition
print("\nPreparing features and target...")
X = final_data[['RIAGENDR', 'RIDAGEYR', 'LBXGLU', 'LBDINSI', 'LBXTC', 'BMI', 'HOMA_IR']]
y = final_data['diabetes_meas']

# Step 3: Handle Missing Values
print("Handling missing values...")
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Step 4: Train-Test Split and Standardization
print("Splitting and scaling data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Model Training with Class Weighting
print("\nCalculating class weights...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Initialize models with class weights
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight=class_weight_dict, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight=class_weight_dict, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),  # Note: GB doesn't support class_weight
    'SVM': SVC(probability=True, class_weight=class_weight_dict, random_state=42)
}

# Step 6: Model Training and Evaluation
print("\nTraining and evaluating models...")
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Find the best performing model
best_model_name = max(results, key=results.get)
print(f"\nBest performing model: {best_model_name} with accuracy: {results[best_model_name]:.3f}")

# Feature importance for tree-based models
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': models[best_model_name].feature_importances_
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values('Importance', ascending=False))

# Step 7: Cross-Validation
print("\nPerforming cross-validation on best model...")
best_model = models[best_model_name]
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})") 