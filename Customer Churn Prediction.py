#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install pandas numpy scikit-learn matplotlib seaborn xgboost


# In[4]:


# Data Manipulation and Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve


# In[17]:


# Load Dataset
df = pd.read_csv('Telco-Customer-Churn.csv')

# Explore Data
print(df.head())


# In[18]:


print(df.info())


# In[19]:


print(df.describe())


# In[24]:


print(df['Churn'].value_counts())
sns.countplot(x=df['Churn'])


# In[ ]:


#Data Preprocessing


# In[6]:


#Handling missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)


# In[21]:


# Convert Binary Categories to 0 and 1
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Encode Other Categorical Features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
categorical_features.remove('customerID')  # Drop customerID as it's not useful

df = pd.get_dummies(df, columns=categorical_features, drop_first=True)


# In[8]:


#Feature Scaling
scaler = StandardScaler()
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numeric_features] = scaler.fit_transform(df[numeric_features])


# In[9]:


# Split Data into Train and Test Sets
X = df.drop(columns=['Churn', 'customerID'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


#Model Building and Training


# In[10]:


#Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions and Evaluation
y_pred_log = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))


# In[11]:


#Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions and Evaluation
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# In[13]:


#XGBoost Classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)

# Predictions and Evaluation
y_pred_xgb = xgb.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))


# In[25]:


#Tenure vs Churn
sns.histplot(df[df['Churn'] == 1]['tenure'], bins=30, color='red', label="Churned", kde=True)
sns.histplot(df[df['Churn'] == 0]['tenure'], bins=30, color='blue', label="Stayed", kde=True)
plt.legend()
plt.title("Tenure Distribution for Churned vs. Retained Customers")
plt.show()


# In[26]:


#Monthly Charges vs. Churn
sns.boxplot(x=df['Churn'], y=df['MonthlyCharges'])
plt.title("Monthly Charges and Churn")
plt.show()


# In[ ]:


#Model Evaluation


# In[14]:


#Confusion Matrix & Classification Report
# Confusion Matrix for XGBoost
conf_matrix = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - XGBoost')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred_xgb))


# In[15]:


#ROC-AUC Curve
y_prob = xgb.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.2f})', color='orange')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve')
plt.legend()
plt.show()


# In[ ]:


#Model Interpretation


# In[16]:


# Feature Importance for XGBoost
importances = xgb.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance - XGBoost')
plt.show()


# In[22]:


import pickle
pickle.dump(xgb, open('churn_model.pkl', 'wb'))


# In[ ]:




