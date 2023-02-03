# Supervised - Credit Risk Analysis

## Objective: 
Implement supervised machine learning on user credit data to train a model which can accurately predict whether a loan would be "High Risk" or "Low Risk".

## Tools & databases used:
* Python
* Pandas
* Scikit-Learn
* Imbalanced-Learn

## Exploration & Cleanup
```
print(loan_stats_df.shape)
(115675, 86)
print(loan_stats_df.isnull().values.any())
True
```
To start I'm inspecting the overall data to gain insight to what I'm handling.
The data size is roughly 115K rows and 86 columns. Of that data I do have null values in here. After looking through column types I see I'm unable to backfill the data. So moving forward I'll be dropping all the null values from every columns.

```
issued_mask = loan_stats_df['loan_status'] != 'Issued'
loan_stats_df = loan_stats_df.loc[issued_mask]
```
Next up, the goal is to evaluate whether our loans are high risk or low risk. So looking further into the "loan_status" column, I see "Issued" loans. Since I'm trying to gain insight into non-issued loans, I can safely drop all rows that have already been finalized.

```
loan_stats_df['int_rate'] = loan_stats_df['int_rate'].str.replace('%', '')
loan_stats_df['int_rate'] = loan_stats_df['int_rate'].astype('float') / 100
```
The "int_rate" column type is a string column so I'm converting that over to a numerical value.

```
low_risk_cols = {'Current': 'low_risk'}   
loan_stats_df = loan_stats_df.replace(low_risk_cols)

high_risk_cols = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'high_risk')    
loan_stats_df = loan_stats_df.replace(high_risk_cols)
loan_stats_df.reset_index(inplace=True, drop=True)
```
For the supervised machine learning I want to predict High Risk and Low Risk, so I'll be grouping the "Current" loans to low risk, and all other lands to high risk.
After doing this the dataset is now very imbalanced with "68,470" low risk, and "347" high risk. Moving forward I'll be looking into using machine learning to handle this imbalance of data.

## Machine Learning: Train Test
```
y = loan_stats_df.loc[:, target].copy()
```
For my "y" value I know that my target is to predict whether a loan is high risk or low risk, so I'll be copying that column over to my "y".

```
X = loan_stats_df.drop(columns='loan_status')
X = pd.get_dummies(X)
```
With my "X" value, no columns stand out as being useless data for my machine learning, so I'm using all other columns as my features. I may need narrow this down in the future, but for now this is my direction. I'm also making sure to convert my string values into numerical values using panda's "get_dummies()" method.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```
With my features and target set, I'm good to test/test the data.

## Machine Learning: Quick Summary
**Accuracy:** Number of predictions that were made correct.  
**Sensitivity (Recall):** The "True Positive Rate", a fraction of the real "true" values that were correctly put into the right bin.  
**Specitivity:** The "True Negative Rate", opposite of Sensitivity and only cares of the "False" cases.  
**Precision:** Of all of values what fraction of the predictions were correct.  
**F1 Score:** The "Harmonic Mean" of recall and precision. Helps give better insight into the performance of imbalanced data.  
**Support:** Tells us how many of each class there were.  
**Confusion Matrix:** Evaluate the performance of the machine learning classification algorithm. [["TP","FP"],["TN","FN"]]  

## Machine Learning: Undersampling
Since the data here is so heavily imbalanced I'll be passing it through a series of balancing methods.

**Balanced Forest Classifier**
This will be my first attempt which will undersample each bootstrap sample. 
```
brfc = BalancedRandomForestClassifier(random_state=1)
brfc.fit(X_train, y_train)
y_pred = brfc.predict(X_test)
0.7885466545953005
confusion_matrix(y_test, y_pred)
[[71,30],[2153,14951]]
print(classification_report_imbalanced(y_test, y_pred))
                  pre       rec       spe        f1       geo       iba       sup
  high_risk       0.03      0.70      0.87      0.06      0.78      0.60       101
   low_risk       1.00      0.87      0.70      0.93      0.78      0.62     17104
avg / total       0.99      0.87      0.70      0.93      0.78      0.62     17205
```
Overall the "BFC" was alright at predictions. As expected it performed very well with low_risk since we have so much data there, while also being terrible at predicting high_risk due to not enough data.

```
importances = brfc.feature_importances_
cols = X.columns
feature_importances_df = pd.DataFrame({'feature':cols, 'importance': importances})
feature_importances_df.loc[feature_importances_df['importance']==0]['feature'].to_list()

['recoveries','collection_recovery_fee','policy_code','acc_now_delinq','delinq_amnt',
 'num_tl_120dpd_2m','num_tl_30dpd','tax_liens','pymnt_plan_n','hardship_flag_N',
 'debt_settlement_flag_N']
```
While I'm still with "BFC" I'm going to take advantage of the "feature_importances_" to gain insight into which features had no impact on the predictions. I'll use this later to attempt to improve predictions.

**Easy Ensemble Classifier**  
This classifier is an ensemble of AdaBoost learners, balancing is achieved with random under-sampling.
```
eec = EasyEnsembleClassifier(n_estimators=100, random_state=1)
eec.fit(X_train, y_train)
y_pred = eec.predict(X_test)
0.9316600714093861
confusion_matrix(y_test, y_pred)
[[93,8],[983,16121]]
print(classification_report_imbalanced(y_test, y_pred))
                   pre       rec       spe        f1       geo       iba       sup
  high_risk       0.09      0.92      0.94      0.16      0.93      0.87       101
   low_risk       1.00      0.94      0.92      0.97      0.93      0.87     17104
avg / total       0.99      0.94      0.92      0.97      0.93      0.87     17205
```
I see improved results with this model, it seems I'm able to predict both high_risk and low_risk much better.

**ClusterCentroids**  
Makes undersampling by generating a new set based on centroids by clustering methods. The algorithm is generating a new set according to the cluster centroid of a KMeans algorithm.
```
cc = ClusterCentroids(random_state=1)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)
0.5447046721744204
confusion_matrix(y_test, y_pred)
[[70,31],[10325,6779]]
print(classification_report_imbalanced(y_test, y_pred))
                   pre       rec       spe        f1       geo       iba       sup
  high_risk       0.01      0.69      0.40      0.01      0.52      0.28       101
   low_risk       1.00      0.40      0.69      0.57      0.52      0.27     17104
avg / total       0.99      0.40      0.69      0.56      0.52      0.27     17205
```
Cluster centroids seems to perform pretty poorly with this dataset. low_risk took a huge hit in scores, overall I can safely rule this out.

## Machine Learning: Oversampling
Next I'm going to explore the accuracy of using oversampling techniques.

**RandomOverSampler**
This will oversample the minority class by picking samples randomly with replacement.
```
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)
0.6463970560994359
confusion_matrix(y_test, y_pred)
[[72,29],[7185,9919]]
print(classification_report_imbalanced(y_test, y_pred))
                   pre       rec       spe        f1       geo       iba       sup
  high_risk       0.01      0.71      0.58      0.02      0.64      0.42       101
   low_risk       1.00      0.58      0.71      0.73      0.64      0.41     17104
avg / total       0.99      0.58      0.71      0.73      0.64      0.41     17205
```
Not much to say about this algorithm, over-sampling this way clearly results in worse scores across the every category outside of precision.

**SMOTE**  
SMOTE works by selecting pair of minority class observations and then creating a synthetic point that lies on the line connecting these two.
```
X_resampled, y_resampled = SMOTE(random_state=1, sampling_strategy='auto').fit_resample(X_train, y_train)
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)
0.6586230769943224
confusion_matrix(y_test, y_pred)
[[64,37],[5412,11692]]
print(classification_report_imbalanced(y_test, y_pred))
                   pre       rec       spe        f1       geo       iba       sup
  high_risk       0.01      0.63      0.68      0.02      0.66      0.43       101
   low_risk       1.00      0.68      0.63      0.81      0.66      0.44     17104
avg / total       0.99      0.68      0.63      0.81      0.66      0.44     17205
```
Using SMOTE I see a mix of better and worse scores. The performance is fairly comparable to the RandomOverSample class.

**SMOTE-ENN**  
This method combines SMOTE oversampling with ENN ability to delete observations.
```
smote_enn = SMOTEENN(random_state=1)
X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
model = LogisticRegression(solver='lbfgs', random_state=1)
y_pred = model.predict(X_test)
0.6361059077142514
confusion_matrix(y_test, y_pred)
[[69,2],[7029,10075]]
print(classification_report_imbalanced(y_test, y_pred))
                   pre       rec       spe        f1       geo       iba       sup
  high_risk       0.01      0.68      0.59      0.02      0.63      0.41       101
   low_risk       1.00      0.59      0.68      0.74      0.63      0.40     17104
avg / total       0.99      0.59      0.68      0.74      0.63      0.40     17205
```
The SMOTE-ENN didn't seem to perform much better than SMOTE. I'm still getting some pretty bad scores with both categories.

## Verdict
After creating 6 different models in order to predict credit risk it seems that Easy Ensemble Classifier is the best model to use for our dataset. ALl around the algorithm performs the best with the data. It's able to predict low_risk very well. The recall score for the high_risk is great and minimized how many high_risk were categorized as low_risk. Since we're having credit loans, having a good recall score for our high_risk cases is very important.
