import numpy as np
import pandas as pd
import seaborn as sns

import os
import matplotlib.pyplot as plt

import sklearn
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from scipy.stats import zscore, pearsonr, uniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from sklearn.linear_model import LogisticRegression



file_path_trainC = "../content/TRAIN_CATEGORICAL_METADATA.xlsx"
train_cat = pd.read_excel(file_path_trainC)
train_cat.head()

train_cat.columns

file_path_trainFCM = "../content/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv"
train_FCM = pd.read_csv(file_path_trainFCM)
train_FCM.head()


train_FCM.columns

file_path_trainQ = "../content/TRAIN_QUANTITATIVE_METADATA.xlsx"
train_Quant = pd.read_excel(file_path_trainQ)
train_Quant.head()

train_Quant.columns

file_path_trainS = "../content/TRAINING_SOLUTIONS.xlsx"
train_Solutions = pd.read_excel(file_path_trainS)
train_Solutions.head()


train_cat.info()

train_cat['Barratt_Barratt_P2_Occ'].value_counts()

train_Quant['MRI_Track_Age_at_Scan'].hist(figsize=(12, 10), bins=20)

train_Solutions['ADHD_Outcome'].value_counts()

train_Solutions['ADHD_Outcome'].value_counts().plot(kind='bar', color='blue')

train_Solutions['Sex_F'].value_counts()

train_Solutions['Sex_F'].value_counts().plot(kind='bar', color='blue')

train_Quant_copy = train_Quant.copy()
train_Quant_copy['ADHD_Outcome'] = train_Solutions['ADHD_Outcome']




train_cat['Barratt_Barratt_P1_Edu'].value_counts()

train_cat_copy = train_cat.copy()
train_cat_copy['ADHD_Outcome'] = train_Solutions['ADHD_Outcome']

adhd_percentages = train_cat_copy.groupby('Barratt_Barratt_P1_Edu')['ADHD_Outcome'].mean()
print(adhd_percentages)

train_cat['Barratt_Barratt_P1_Edu'].value_counts()

for col in train_cat.select_dtypes(include='int').columns:
    train_cat[col] = train_cat[col].astype('category')

    # Creating a list of all of the columns except the first
columns_to_encode = train_cat.columns[1:].tolist()

# Print the columns to encode
print("Columns to encode:", columns_to_encode)

train_encoded = pd.get_dummies(train_cat[columns_to_encode], drop_first=True)
train_encoded = train_encoded.applymap(lambda x: 1 if x is True else (0 if x is False else x))

cat_train_final = pd.concat([train_cat.drop(columns=columns_to_encode), train_encoded], axis=1)

# ensure it looks correct
cat_train_final.head()

file_path_testC = "../content/TEST_CATEGORICAL.xlsx"
test_cat = pd.read_excel(file_path_testC)
#(test_cat.head()


for col in test_cat.select_dtypes(include='int').columns:
    test_cat[col] = test_cat[col].astype('category')

# Encode categorical variables in test
test_encoded = pd.get_dummies(test_cat[columns_to_encode], drop_first=True)
test_encoded = test_encoded.applymap(lambda x: 1 if x is True else (0 if x is False else x))

# Ensure test_encoded has the same columns as train_encoded
missing_cols = set(train_encoded.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0  # Add missing columns with 0 values

# Ensure test_encoded columns are in the same order as train_encoded
test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

# Combine encoded columns with the rest of the DataFrame
cat_test_final = pd.concat([test_cat.drop(columns=columns_to_encode), test_encoded], axis=1)

cat_test_final.head()

train_cat_FCM = pd.merge(cat_train_final, train_FCM, on = 'participant_id')

train_df = pd.merge(train_cat_FCM, train_Quant, on = 'participant_id')

print(train_df)

# ensure it looks accurate
train_df.head()

print(train_df.to_csv)

file_path_testFCM = "../content/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv"
test_FCM = pd.read_csv(file_path_testFCM)
#print(train_FCM.head())

file_path_testQ = "../content/TEST_QUANTITATIVE_METADATA.xlsx"
test_Quant = pd.read_excel(file_path_testQ)
#print(train_Quant.head())

test_cat_FCM = pd.merge(cat_test_final, test_FCM, on = 'participant_id')

test_df = pd.merge(test_cat_FCM, test_Quant, on = 'participant_id')

# ensure it looks accurate
test_df.head()

# check how many NA values we have
print(train_df.isna().sum())

# 371 NANs values
# 360 in MRI_Track_age_at_Scan
# 11 in PreInt_Demos_Fam_Child_Ethnicity


train_df.fillna({'MRI_Track_Age_at_Scan':train_df['MRI_Track_Age_at_Scan'].mean()}, inplace = True)
train_df.fillna({'PreInt_Demos_Fam_Child_Ethnicity':train_df['PreInt_Demos_Fam_Child_Ethnicity'].mean()}, inplace = True)

print(train_df.isna().sum().sum()) # should now be zero

train_df.ffill(inplace=True)
print(train_df.isna().sum().sum())

for col in test_df.columns:
    if test_df[col].isna().sum() > 0:  # Check if the column has NaN values
        if test_df[col].dtype in ['float64', 'int64']:  # Ensure it's numeric
            test_df[col] = test_df[col].fillna(test_df[col].mean())  # Avoid inplace
        else:
            print(f"Skipping non-numeric column: {col}")


from sklearn.preprocessing import MinMaxScaler

# Ensure train_Quant is a DataFrame
import pandas as pd
# Example: train_Quant = pd.read_csv("your_data.csv")  # Load your data

# Define the column to scale
column_to_normalize = "EHQ_EHQ_Total"
column = train_Quant[column_to_normalize].values.reshape(-1, 1)
print("Column waiting to be normalized")
print(column)

######### Initialize the MinMaxScaler #########
scaler = MinMaxScaler()

# Normalize the data
normalized_data = scaler.fit_transform(column)

# print("Normalized column")
# print(normalized_data)


######### Or with a custom range (-1, 1) #########
scaler = MinMaxScaler(feature_range=(-1, 1))

# Normalize the data
normalized_data = scaler.fit_transform(column)

#print("Normalized column")
#print(normalized_data)

######### Manual scaling (alternative method) #########
min_val = train_Quant[column_to_normalize].min()
max_val = train_Quant[column_to_normalize].max()

# Handle the case where all values are the same
if min_val == max_val:
    column_copy = 0.0  # Assign a constant if all values are the same
else:
    column_copy = (column - min_val) / (max_val - min_val)

# print("Normalized column")
# print(column_copy)


file_path_trainS = "../content/TRAINING_SOLUTIONS.xlsx"
train_Solutions = pd.read_excel(file_path_trainS)

X_train = train_df.drop(columns = ['participant_id'])
Y_train = train_Solutions.drop(columns = ['participant_id'])



# Initialize the base classifier
xgb_classifier = XGBClassifier(objective='binary:logistic', n_estimators=200, learning_rate=0.01, max_depth=5)

# Wrap with MultiOutputClassifier for multi-target classification
multioutput_classifier = MultiOutputClassifier(xgb_classifier)

# Train the model


multioutput_classifier.fit(X_train, Y_train)


participant_id = test_df['participant_id']

X_test = test_df.drop(columns = 'participant_id')

y_pred = multioutput_classifier.predict(X_test)

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(
    y_pred,
    columns=['Predicted_Gender', 'Predicted_ADHD']
)

# Combine participant IDs with predictions
result_df = pd.concat([participant_id.reset_index(drop=True), predictions_df], axis=1)

# To add a save result_df
result_df.to_excel("upload_to_kaggle.xlsx", index=False)
result_df.to_csv("upload_to_kaggle.csv", index=False)
# Print or save the DataFrame
print(result_df)

# def multi_output_accuracy(y_true, y_pred):
#     # Ensure y_true and y_pred are NumPy arrays
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     # Compute accuracy for each target variable and return the mean
#     return np.mean([accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])

# multi_output_scorer = make_scorer(multi_output_accuracy)


# # Perform cross-validation on the training data
# cv_scores = cross_val_score(multioutput_classifier, X_train, Y_train, cv=5, scoring=multi_output_scorer)

# # Output the cross-validation results
# print("Cross-validation scores for each fold:", cv_scores)
# print("Mean CV score:", np.mean(cv_scores))

# # Cross-validation scores for each fold: [0.82304527 0.78600823 0.69341564 0.64669421 0.33471074]
# # Mean CV score: 0.6567748188960311


# model = LogisticRegression(max_iter=1000)
# model.fit(train_df.drop(columns='participant_id'), train_Solutions['Sex_F'])


# # Get coefficients for Sex prediction
# coefficients = pd.Series(model.coef_[0], index=train_df.drop(columns='participant_id').columns)


# # Select top features for Sex prediction
# top_features = coefficients.abs().nlargest(10)
# print(top_features)


# #Plotting the top 10 coefficents for Sex Outcome
# plt.figure(figsize=(10,6))
# top_features.sort_values().plot(kind='barh', color='skyblue')
# plt.title('Top 10 Features for Sex Outcome')
# plt.ylabel('Features')
# plt.xlabel('Absolute Coefficient Value')
# plt.xticks(rotation=45, ha='right')
# plt.show()


# model = LogisticRegression(max_iter=1000)
# model.fit(train_df.drop(columns='participant_id'), train_Solutions['ADHD_Outcome'])

# # Get coefficients for ADHD_Outcome prediction
# coefficients = pd.Series(model.coef_[0], index=train_df.drop(columns='participant_id').columns)

# # Select top features for ADHD_Outcome prediction
# top_features = coefficients.abs().nlargest(10)
# print(top_features)


# #Plotting the top 10 coefficents
# plt.figure(figsize=(10,6))
# top_features.sort_values().plot(kind='barh', color='skyblue')
# plt.title('Top 10 Features for ADHD Outcome')
# plt.ylabel('Features')
# plt.xlabel('Absolute Coefficient Value')
# plt.xticks(rotation=45, ha='right')
# plt.show()

# model = LogisticRegression(penalty='l1', solver='liblinear')
# model.fit(train_df.drop(columns='participant_id'), train_Solutions['Sex_F'])

# selected_features = train_df.drop(columns='participant_id').columns[model.coef_[0] != 0]
# print(selected_features)

# model = LogisticRegression(penalty='l1', solver='liblinear')
# model.fit(train_df.drop(columns='participant_id'), train_Solutions['ADHD_Outcome'])

# selected_features = train_df.drop(columns='participant_id').columns[model.coef_[0] != 0]
# print(selected_features)

# # Step 1: Find common features between ADHD and Sex selected features
# common_features = list(set(selected_features_ADHD) or set(selected_features_Sex))


# X_train_2 = X_train[common_features]
# X_test_2 = X_test[common_features]

# # Initialize the base classifier
# xgb_classifier = XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=5)

# # Wrap with MultiOutputClassifier for multi-target classification
# multioutput_classifier = MultiOutputClassifier(xgb_classifier)

# # Train the model
# multioutput_classifier.fit(X_train_2, Y_train)

# y_pred_2 = multioutput_classifier.predict(X_test_2)

# # Convert predictions to a DataFrame
# predictions_df_2 = pd.DataFrame(
#     y_pred_2,
#     columns=['Predicted_Gender', 'Predicted_ADHD']
# )

# # Combine participant IDs with predictions
# result_df_2 = pd.concat([participant_id.reset_index(drop=True), predictions_df_2], axis=1)

# result_df_2.head()

# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import make_scorer, accuracy_score

# # Create a scorer using scikit-learn's make_scorer
# multi_output_scorer = make_scorer(multi_output_accuracy)

# # Perform cross-validation on the training data
# cv_scores_2 = cross_val_score(multioutput_classifier, X_train_2, Y_train, cv=5, scoring=multi_output_scorer)

# # Output the cross-validation results
# print("Cross-validation scores for each fold:", cv_scores_2)
# print("Mean CV score:", np.mean(cv_scores_2))

# #Cross-validation scores for each fold: [0.79423868 0.79218107 0.71193416 0.69834711 0.39669421]
# #Mean CV score: 0.678679046355814
