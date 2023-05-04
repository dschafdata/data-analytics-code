import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
medical = pd.read_csv('C:/users/dscha/Downloads/D208/medical_clean.csv')

pd.DataFrame.duplicated(medical) #check for duplicate data
pd.DataFrame.isnull(medical).sum() #check for nulls

from sklearn import linear_model
from sklearn.metrics import roc_auc_score

def auc(variables, target, basetable): #Create auc function
 X = basetable[variables]
 y = basetable[target]
 logreg = linear_model.LogisticRegression(max_iter = 1000)
 logreg.fit(X, y.values.ravel())
 predictions = logreg.predict_proba(X)[:,1]
 auc = roc_auc_score(y, predictions)
 return(auc)

medical['HighBlood_numeric'] = medical['HighBlood'] 
dict_hbn = {"HighBlood_numeric": {"No":0,"Yes":1}} 
medical.replace(dict_hbn, inplace=True) 
medical['Stroke_numeric'] = medical['Stroke'] 
dict_srk = {"Stroke_numeric": {"No":0,"Yes":1}}
medical.replace(dict_srk, inplace=True)
medical['Arthritis_numeric'] = medical['Arthritis']
dict_art = {"Arthritis_numeric": {"No":0,"Yes":1}}
medical.replace(dict_art, inplace=True)
medical['Diabetes_numeric'] = medical['Diabetes']
dict_dia = {"Diabetes_numeric": {"No":0,"Yes":1}}
medical.replace(dict_dia, inplace=True)
medical['Hyperlipidemia_numeric'] = medical['Hyperlipidemia']
dict_hln = {"Hyperlipidemia_numeric": {"No":0,"Yes":1}}
medical.replace(dict_hln, inplace=True)
medical['BackPain_numeric'] = medical['BackPain']
dict_bpn = {"BackPain_numeric": {"No":0,"Yes":1}}
medical.replace(dict_bpn, inplace=True)
medical['Allergic_rhinitis_numeric'] = medical['Allergic_rhinitis']
dict_arn = {"Allergic_rhinitis_numeric": {"No":0,"Yes":1}}
medical.replace(dict_arn, inplace=True)
medical['Reflux_esophagitis_numeric'] = medical['Reflux_esophagitis']
dict_ren = {"Reflux_esophagitis_numeric": {"No":0,"Yes":1}}
medical.replace(dict_ren, inplace=True)
medical['Asthma_numeric'] = medical['Asthma']
dict_ast = {"Asthma_numeric": {"No":0,"Yes":1}}
medical.replace(dict_ast, inplace=True)
medical['ReAdmis_numeric'] = medical['ReAdmis']
dict_tar = {"ReAdmis_numeric": {"No":0,"Yes":1}}
medical.replace(dict_tar, inplace=True)
medical['Overweight_numeric'] = medical['Overweight']
dict_own = {"Overweight_numeric": {"No":0,"Yes":1}}
medical.replace(dict_own, inplace=True)
ohe_df = pd.get_dummies(medical['Initial_admin']) #One-hot encode the Initial Admin column
medical = pd.concat([medical,ohe_df], axis=1) #add the ohe_df to the end of the medical dataframe
medclean = medical[['ReAdmis_numeric', 'HighBlood_numeric', 'Stroke_numeric', 'Arthritis_numeric',
                    'Diabetes_numeric', 'Hyperlipdemia_numeric','BackPain_numeric','Allergic_rhinitis_numeric',
                    'Reflux_esophagitis_numeric', 'Asthma_numeric', 'Overweight_numeric', 'Observation Admission',
                    'Emergency Admission', 'Initial_days']] #Create a new dataframe for logistic regression, leaving Elective Admission off to account
                    #for the One-Hot Encoding/Dummy Variable trap.

logreg=linear_model.LogisticRegression()
X= medclean.drop('ReAdmis_numeric', axis=1)
y= medclean[["ReAdmis_numeric"]]


import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

#Use backward stepwise method of reducing the features used
auc_b1 = auc(['Initial_days','HighBlood_numeric', 'Stroke_numeric','Arthritis_numeric', 'Diabetes_numeric','Asthma_numeric',
              'Emergency Admission', 'Observation Admission'],['ReAdmis_numeric'], medclean)              
print(auc_b1) #Result of 0.99898

auc_b2 = auc(['Initial_days','HighBlood_numeric', 'Stroke_numeric','Arthritis_numeric', 'Diabetes_numeric',
              'Emergency Admission', 'Observation Admission'],['ReAdmis_numeric'], medclean)              
print(auc_b2) #Result of 0.99890
auc_b3 = auc(['Initial_days','HighBlood_numeric', 'Stroke_numeric','Arthritis_numeric',
              'Emergency Admission', 'Observation Admission'],['ReAdmis_numeric'], medclean)              
print(auc_b3) #Result of 0.99888

auc_b4 = auc(['Initial_days','HighBlood_numeric', 'Stroke_numeric','Emergency Admission', 'Observation Admission'],['ReAdmis_numeric'], medclean)              
print(auc_b4) #Result of 0.9988

auc_b5 = auc(['Initial_days','HighBlood_numeric','Emergency Admission', 'Observation Admission'],['ReAdmis_numeric'], medclean)              
print(auc_b5) #Result of 0.9987

auc_b6 = auc(['Initial_days','Emergency Admission', 'Observation Admission'],['ReAdmis_numeric'], medclean)              
print(auc_b6) #Result of 0.9986

auc_b7 = auc(['Initial_days','Emergency Admission'],['ReAdmis_numeric'], medclean)              
print(auc_b7) #Result of 0.9986

auc_b8 = auc(['Initial_days'],['ReAdmis_numeric'], medclean)              
print(auc_b8) #Result of 0.9984

X_refined = medclean[["Initial_days"]]
y_refined = medclean[["ReAdmis_numeric"]]
logit_model=sm.Logit(y_refined,X_refined)
result=logit_model.fit()
print(result.summary2())


logreg.fit(X_refined,y_refined.values.ravel())
print(logreg.coef_)
print(logreg.intercept_)

def auc_train_test(variables, target, train, test):
 X_train = train[variables]
 X_test = test[variables]
 Y_train = train[target]
 Y_test = test[target]
 logreg = linear_model.LogisticRegression()
 
 # Fit the model on train data
 logreg.fit(X_train, Y_train.values.ravel())
 
 # Calculate the predictions both on train and test data
 predictions_train = logreg.predict_proba(X_train)[:,1]
 predictions_test = logreg.predict_proba(X_test)[:,1]
  # Calculate the AUC both on train and test data
 auc_train = roc_auc_score(Y_train, predictions_train)
 auc_test = roc_auc_score(Y_test,predictions_test)
 return(auc_train, auc_test)


from statsmodels.formula.api import logit

mdl_medical= logit("ReAdmis_numeric ~ Initial_days", data=medclean).fit()

conf_matrix = mdl_medical.pred_table()

print(conf_matrix)


tp = conf_matrix[1,1]
tn = conf_matrix[0,0]
fp = conf_matrix[0,1]
fn = conf_matrix[1,0]
print(tp, tn, fp, fn)

accuracy = (tp+tn)/(tp+tn+fp+fn)
misclass = 1 -accuracy
precision = tp/(tp+fn)
recall = tp/(tp+fp)
specificity = tn/(tn+fp)
f'The Accuracy is {accuracy}, the missclassification is {misclass}, the precision is {precision}, the recall is {recall}, and the specificity is {specificity}'
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_refined, y_refined, test_size=.5, stratify = Y)


train = pd.concat([X_train, y_train], axis=1)
test = pd.concat ([X_test, y_test],axis=1)

auc_train, auc_test = auc_train_test(["Initial_days"],["ReAdmis_numeric"], train, test)
print(round(auc_train,3)) #Result was 0.999
print(round(auc_test,3)) #Result was 0.998
sns.regplot (data=medclean,x="Initial_days", y="ReAdmis_numeric", ci=None, logistic=True)
explanatory_data=pd.DataFrame({"Initial_days": np.arange(0,70,1)})
prediction_data = explanatory_data.assign(ReAdmis_numeric = mdl_medical.predict(explananatory_data))

sns.regplot(x='Initial_days',y='ReAdmis_numeric', data=medical, ci=None, logistic=True)
sns.scatterplot(x='Initial_days',y='ReAdmis_numeric', data=prediction_data, color='red')
plt.show()


