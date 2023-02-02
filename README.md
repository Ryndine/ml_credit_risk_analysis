# Supervised - Credit Risk Analysis

## Objective: 
Implement machine learning models using resampling and ensemble to address class imbalance, and evaluate the performance of different machine learning models.

## Tools & databases used:
- Python, Sci-kit Learn, Imbalanced Learn

## Cleaning & Setup:
Prior to running machine learning I need to cleanup the dataset.
- Set target, loan_status for machine learning
- Dropping null values.
- Drop Issued column since it's unneeded.
- Reformat numerical data.
- Convert loan_status from Current to low_risk & (Late/Default/Grace) to high_risk.
```
# Drop the null columns where all values are null
loan_stats_df = loan_stats_df.dropna(axis='columns', how='all')
loan_stats_df = loan_stats_df.dropna()

# Don't need Issued column
issued_mask = loan_stats_df['loan_status'] != 'Issued'
loan_stats_df = loan_stats_df.loc[issued_mask]

# Numerical interest rate
loan_stats_df['int_rate'] = loan_stats_df['int_rate'].str.replace('%', '')
loan_stats_df['int_rate'] = loan_stats_df['int_rate'].astype('float') / 100


# convert columns to low_risk and high_risk based on their values
x = {'Current': 'low_risk'}   
loan_stats_df = loan_stats_df.replace(x)

x = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'high_risk')    
loan_stats_df = loan_stats_df.replace(x)
loan_stats_df.reset_index(inplace=True, drop=True)
```

## Analysis

**Train & Test**  
Now I want to plug the data into training and testing data.
```
X = loan_stats_df.drop(columns='loan_status')
X = pd.get_dummies(X)
y = loan_stats_df.loc[:, target].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```

**Balanced Forest Classifier**  
For the balanced forest model I ended up with an accuracy score of .78
![bfc_accuracy](/Resources/Images/bfc_accuracy.jpg)

**Easy Ensemble AdaBoost Classifier**  
Using AdaBoost I saw great improvement with an accuracy score of .93
![adaboost_accuracy](/Resources/Images/adaboost_accuracy.jpg)

**Naive Random Oversampling**  
With Naive Random Oversampling I saw a decrease in accuracy with a score of .64
![oversample_accuracy](/Resources/Images/oversample_accuracy.jpg)

**SMOTE**  
SMOTE oversample received a similar accuracy score as Naive Random at .66
![smote_accuracy](/Resources/Images/smote_accuracy.jpg)

**Custer Centroid**  
The worst performing model so far with an accuracy score of .52
![ccentroid_accuracy](/Resources/Images/ccentroid_accuracy.jpg)

**SMOTEENN**  
This resampling performed similarly to SMOTE and Naive oersamples with an accuracy of .63
![smoteenn_accuracy](/Resources/Images/smoteenn_accuracy.jpg)

## Verdict

After creating 6 different models in order to predict credit risk it seems that Easy Ensemble AdaBoost Classifier is the best model to use for our dataset. It has a high recall and high precision score which means it's picky but very accurate. The accuracy score reflects that with a great score of .93.
