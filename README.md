import sys
print("Python version")
print (sys.version)
print("Version info.")
print (sys.version_info)

print()
import datetime
now = datetime.datetime.now()
print("current date and time : ")
print(now.strftime("%Y-%M-%D %H-%M-%S"))


import pandas as pd
import numpy as np

adult_df = pd.read_csv(r"C:\akshay\Python\adult.data" , header = None ,delimiter =" *, *" , engine = "python") ##in ouput column name is missing
adult_df.head()

adult_df.shape

adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']                               ##assign the column name which are intital they are blank

adult_df.head()

#creating the copy of dataframe

adult_df_rev = pd.DataFrame.copy(adult_df)

#adult_df_rev.describe(include ="all")

##we are removing the education col because it represnet with education_num which is same and fnlwgt is removed becasue it dose not have any depedancy on income
adult_df_rev = adult_df_rev.drop(["education" , "fnlwgt"],axis =1)

adult_df_rev.isnull().sum()

adult_df_rev = adult_df_rev.replace(["?"] , np.nan)
adult_df_rev.isnull().sum()

#replace missing value in mode values workplace,occupation and native_country has the missing value

for value in["workclass", "occupation" ,"native_country"]:
    adult_df_rev[value].fillna(adult_df_rev[value].mode()[0],inplace = True)
    
adult_df_rev.workclass.mode()

adult_df_rev.workclass
adult_df_rev.head()

#creating the list of categorical data
colname = ["workclass",
            "marital_status","occupation",
            "relationship","race","sex",
            "native_country","income"]
colname

## Preproccessig of data using sklearn library/Converting cat data to numeric


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for x in colname:
    adult_df_rev[x] =le.fit_transform(adult_df_rev[x])

adult_df_rev.head()

    # 0---> <-50k
# 1---> >-50k

adult_df_rev.dtypes


x=adult_df_rev.values[: , 0:-1]
y=adult_df_rev.values[: , -1]


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(x)

x =scaler.transform(x)
print(x)


#converting y variable in categorical data(in this case we have already convert y in categgorical data)
y = y.astype(int)
y

#Running the basic model

from sklearn.model_selection import train_test_split
#split the data into tesst and train
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state =10 )

# we have follow this syntax alll over the project.Random_state used beacause generate same output
# bydefault sklearn has it own splitting ratio as 75% to 25%

x_train

#creating the model
from sklearn.linear_model import LogisticRegression

#creating model
classifier = LogisticRegression()
#fitting training data to model
classifier.fit(x_train,y_train) #fit function used to train the model
y_pred = classifier.predict(x_test)
print(list(zip(y_test,y_pred)))  ##to campare the y actual to y pred
#print(classifier.coef_)
#print(classifier.intercept_)

#adjusting the Threshold
#STORE THE PREDICTED PROBALITIIEES
y_pred_prob = classifier.predict_proba(x_test)
y_pred_prob

y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value > 0.45:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print (y_pred_class)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(y_test,y_pred_class)
print(cfm)

print("Classification report:")

print(classification_report(y_test,y_pred_class))

acc=accuracy_score(y_test,y_pred_class)
print("Accuracy of the modal:",acc)

for a in np.arange(0,1,0.05):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", 
    cfm[1,0]," , type 1 error:", cfm[0,1])
    
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(y_test,y_pred)
print(cfm)

print("Classification report:")

print(classification_report(y_test,y_pred))

acc=accuracy_score(y_test,y_pred)
print("Accuracy of the modal:",acc)


#ROC and AUC curve implementing

## AUC value and ROC curev implementing

from sklearn import metrics
fpr, tpr, z = metrics.roc_curve(y_test, y_pred_prob[:,1])
auc = metrics.auc(fpr,tpr)
print(auc)
#print(z)
#and ROC curev implementing
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')   
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')  ## legend fuction use to place the 
plt.plot([0, 1], [0, 1],'r--') ##start poinit of x and y axix
plt.xlim([0, 1]) #range of x axis
plt.ylim([0, 1]) #range of y axis
plt.xlabel('False Positive Rate') #label to axix
plt.ylabel('True Positive Rate') #label to axix
plt.show() ## to show only



## AUC value and ROC curev implementing base on threshold of 0.46
from sklearn import metrics
fpr, tpr, z = metrics.roc_curve(y_test, y_pred_class)
auc = metrics.auc(fpr,tpr)
print(auc)
print(fpr)
print(tpr)
#print(z)


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')   
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')  ## legend fuction use to place the 
plt.plot([0, 1], [0, 1],'r--') ##start poinit of x and y axix
plt.xlim([0, 1]) #range of x axis
plt.ylim([0, 1]) #range of y axis
plt.xlabel('False Positive Rate') #label to axix
plt.ylabel('True Positive Rate') #label to axix

plt.show() ## to show only

#Using cross validation

classifier=(LogisticRegression())   ##change as per your selected algo 

#performing kfold_cross_validation
from sklearn.model_selection import KFold     
kfold_cv=KFold(n_splits=10)  ##creating cross validation model  n_splits=10 is argument
print(kfold_cv)

from sklearn.model_selection import cross_val_score ##responsible for th 10 iteration
#running the model using scoring metric as accuracy
kfold_cv_result=cross_val_score(estimator=classifier,X=x_train,
y=y_train, cv=kfold_cv)     ##use cross_val_score function
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())

#_______________________________________________________

for train_value, test_value in kfold_cv.split(x_train):
    classifier.fit(x_train[train_value], y_train[train_value]).predict(x_train[test_value])


Y_pred=classifier.predict(x_test)
#print(list(zip(Y_test,Y_pred)))

# comparing the accuracy of model with cross tabulatiion accuracy

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(y_test,Y_pred)
print(cfm)

print("Classification report:")

print(classification_report(y_test,Y_pred))

acc=accuracy_score(y_test,Y_pred)
print("Accuracy of the modal:",acc)
