import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv("train.csv")


###############################################################################################################
##############################----------Step 1::Data statistics-------------###################################
###############################################################################################################

#Data statistics-> Shape, Datatypes ,Description ,Skew, Class distribution

#---------------------------------------------------------------------------------------------------------------#
#Shape
train.shape     
# Learning: Data is loaded successfully as dimensions match the data description
#---------------------------------------------------------------------------------------------------------------#
#Datatypes
train.dtypes    
# Learning: Data types of all attributes has been inferred as int64
#---------------------------------------------------------------------------------------------------------------#
#Description
pd.set_option('display.max_columns', None)
train.describe() 
# Learning :
# No attribute is missing as count is 15120 for all attributes. Hence, all rows can be used
# Negative value(s) present in Vertical_Distance_To_Hydrology. Hence, some tests such as chi-sq cannot be used.
# Wilderness_Area and Soil_Type are one hot encoded. Hence, they could be converted back for some analysis
# Attributes Soil_Type7 and Soil_Type15 can be removed as they are constant
# Scales are not the same for all. Hence, rescaling and standardization may be necessary for some algos
#---------------------------------------------------------------------------------------------------------------#
#Skew
train.skew() 
# Values close to 0 show less skew
# Several attributes in Soil_Type show a large skew. Hence, some algos may benefit if skew is corrected or delete from analysis

###############################################################################################################
##############################----------End of Step 1::Data statistics-------------############################
###############################################################################################################



###############################################################################################################
##############################----------Step 2::Data Interaction-------------##################################
###############################################################################################################

#Data Interaction 1.Correlation 2.Scatter plot

#---------------------------------------------------------------------------------------------------------------#
#https://www.kaggle.com/sharmasanthosh/exploratory-study-on-feature-selection
# Correlation 
# Correlation tells relation between two attributes. Correlation requires continous data. Hence, ignore Wilderness_Area and Soil_Type as they are binary

# Anand To do: Develop visual Highest correlated pair function

train.corr(method='pearson', min_periods=1)
f,ax = plt.subplots(figsize=(30, 30))
sns.heatmap(train.corr(), annot=True, linewidths=1, fmt= '.1f',ax=ax)


#sets the number of features considered
size = 11 

#create a dataframe with only 'size' features
data=train.iloc[:,:size] 

#get the names of all the columns
cols=data.columns 

# Calculates pearson co-efficient for all combinations
data_corr = data.corr()

# Set the threshold to select only only highly correlated attributes
threshold = 0.5

# List of pairs along with correlation above threshold
corr_list = []

#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))


#correlation map
#---------------------------------------------------------------------------------------------------------------#
# Scatter plot 



###############################################################################################################
##############################----------End Step 2::Data Interaction-------------##############################
###############################################################################################################
    

###############################################################################################################
##############################----------Step 3::Data Visualization-------------##################################
###############################################################################################################

#Data Visualization 1. Box and density plots 2.Grouping of one hot encoded attributes
    
#---------------------------------------------------------------------------------------------------------------#
# Box and density plots
# We will visualize all the attributes using Violin Plot - a combination of box and density plots

#names of all the attributes 
cols = train.columns

#number of attributes (exclude target)
size = len(cols)-1

#x-axis has target attribute to distinguish between classes
x = cols[size]

#y-axis shows values of an attribute
y = cols[0:size]

#Plot violin for all attributes
for i in range(1,size):
    sns.violinplot(data=train,x=x,y=y[i])  
    plt.show()
#    sns.boxplot(data=train,x=x,y=y[i])  
#    plt.show()
#---------------------------------------------------------------------------------------------------------------#    
#Grouping of one hot encoded attributes
    
#names of all the columns
dataset=train
cols = dataset.columns

#number of rows=r , number of columns=c
r,c = dataset.shape

#Create a new dataframe with r rows, one column for each encoded category, and target in the end
data = pd.DataFrame(index=np.arange(0, r),columns=['Wilderness_Area','Soil_Type','Cover_Type'])

#Make an entry in 'data' for each r as category_id, target value
for i in range(0,r):
    w=0;
    s=0;
    # Category1 range
    for j in range(10,14):
        if (dataset.iloc[i,j] == 1):
            w=j-9  #category class
            break
    # Category2 range        
    for k in range(14,54):
        if (dataset.iloc[i,k] == 1):
            s=k-13 #category class
            break
    #Make an entry in 'data' for each r as category_id, target value        
    data.iloc[i]=[w,s,dataset.iloc[i,c-1]]

#Plot for Category1    
plt.rc("figure", figsize=(15, 8))
sns.countplot(x="Wilderness_Area", hue="Cover_Type", data=data)
plt.show()
#Plot for Category2
plt.rc("figure", figsize=(25, 10))
sns.countplot(x="Soil_Type", hue="Cover_Type", data=data)
plt.show()

#(right-click and open the image in a new window for larger size)
#WildernessArea_4 has a lot of presence for cover_type 4. Good class distinction
#WildernessArea_3 has not much class distinction
#SoilType 1-6,10-14,17, 22-23, 29-33,35,38-40 offer lot of class distinction as counts for some are very hig
    
    
    
    

#Step 4
#Data Cleaning
    #Remove unnecessary columns

#Step 5
#Data Preparation
    #Original
    #Delete rows or impute values in case of missing
    #StandardScaler
    #MinMaxScaler
    #Normalizer

#Step 6
#Feature selection
    #ExtraTreesClassifier
    #GradientBoostingClassifier
    #RandomForestClassifier
    #XGBClassifier
    #RFE
    #SelectPercentile
    #PCA
    #PCA + SelectPercentile
    #Feature Engineering

#Step 7
#https://www.kaggle.com/sharmasanthosh/exploratory-study-of-ml-algorithms
#Evaluation, prediction, and analysis
    #LDA (Linear algo)
    #LR (Linear algo)
    #KNN (Non-linear algo)
    #CART (Non-linear algo)
    #Naive Bayes (Non-linear algo)
    #SVC (Non-linear algo)
    #Bagged Decision Trees (Bagging)
    #Random Forest (Bagging)
    #Extra Trees (Bagging)
    #AdaBoost (Boosting)
    #Stochastic Gradient Boosting (Boosting)
    #Voting Classifier (Voting)
    #MLP (Deep Learning)
    #XGBoost



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data Exploration

train=pd.read_csv("train.csv")

#Observatios
# 1. No missing data
# 2. All continous Variables 
# 3. Check the Data distribution 
# 4. Elevation is not Normally distributed
# 5. Without doing anything Feature selection/Engineering the base accuracy is 83.5% " Lets see if this can be imporved"



sns.distplot(train['Elevation'] & train['Cover_Type']==1 )
sns.distplot(train['Hillshade_Noon'])
sns.jointplot(x=train['Elevation'], y=train['Aspect']);
sns.pairplot(train)


train.corr(method='pearson', min_periods=1)

#correlation map
f,ax = plt.subplots(figsize=(30, 30))
sns.heatmap(train.corr(), annot=True, linewidths=1, fmt= '.1f',ax=ax)

######################################################################################################
# lets check the normal prediction 
data = pd.read_csv('train.csv')
X= data.loc[:,(data.columns != 'Cover_Type') 
#             & (data.columns != 'Soil_Type8') 
#             & (data.columns != 'Soil_Type25')  
#             & (data.columns != 'Soil_Type28')  
#             & (data.columns != 'Soil_Type9')  
#             & (data.columns != 'Soil_Type36') 
#             & (data.columns != 'Soil_Type27')  
#             & (data.columns != 'Soil_Type21')
#             & (data.columns != 'Soil_Type34')
#             & (data.columns != 'Soil_Type37')
#             & (data.columns != 'Soil_Type19')
#             & (data.columns != 'Soil_Type26')
#             & (data.columns != 'Soil_Type18')
#             & (data.columns != 'Soil_Type35')
#             & (data.columns != 'Soil_Type16')
#             & (data.columns != 'Soil_Type20')
             ]
y =data.loc[:,'Cover_Type']


#Soil_Type8
#Soil_Type25
#Soil_Type28
#Soil_Type9
#Soil_Type36
#Soil_Type27
#Soil_Type21


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#----------------------------------------------------------------------------
# Logistic Regression
# Learning: Logistic Regression 68% of Accuracy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('train.csv')
X= data.loc[:,(data.columns != 'Cover_Type') ]
y =data.loc[:,'Cover_Type']

# Splitting the dataset into the Training set and Test set
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, classification_report
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred_prob = classifier.predict_proba(X_test)[:,1]
print(accuracy_score(y_test, y_pred))
#cm = confusion_matrix(y_test,y_pred)
#print('Confusion matrix: \n',cm)
print('Classification report: \n',classification_report(y_test,y_pred))


#----------------------------------------------------------------------------
# Ramdom Forest
# Learning: Ramdom Forest provides 82% of Accuracy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('train.csv')
X= data.loc[:, (data.columns != 'Cover_Type')
               & (data.columns != 'Id')
             & (data.columns != 'Soil_Type7')
             & (data.columns != 'Soil_Type8')
             & (data.columns != 'Soil_Type9')
             & (data.columns != 'Soil_Type15')
             & (data.columns != 'Soil_Type34')
             & (data.columns != 'Soil_Type36')
             & (data.columns != 'Soil_Type37')
 ]
#X= data[['Elevation'
#         ,'Horizontal_Distance_To_Roadways'
#         ,'Horizontal_Distance_To_Fire_Points'
#         ,'Vertical_Distance_To_Hydrology'
#        ,'Aspect'
#        ,'Hillshade_3pm'
#        ,'Hillshade_9am'
#        ,'Hillshade_Noon'
#        ,'Horizontal_Distance_To_Hydrology'
#        ,'Slope'
#        ,'Soil_Type39'
#        ,'Soil_Type38'
#        ,'Wilderness_Area1'
#        ,'Soil_Type10'
#        ,'Soil_Type3'
#        ,'Soil_Type23'
#        ,'Wilderness_Area3'
#        ,'Wilderness_Area2'
#        ,'Wilderness_Area4'
#        ,'Soil_Type33'
#        ,'Soil_Type13'
#        ,'Soil_Type32'
#        ,'Soil_Type17'
#        ,'Soil_Type4'
#        ,'Soil_Type40'
#        ,'Soil_Type22'
#        ,'Soil_Type2'
#
#         
#         ] ]
y =data.loc[:,'Cover_Type']

# Splitting the dataset into the Training set and Test set
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, classification_report
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred_prob = classifier.predict_proba(X_test)[:,1]
print(accuracy_score(y_test, y_pred))
#cm = confusion_matrix(y_test,y_pred)
#print('Confusion matrix: \n',cm)
print('Classification report: \n',classification_report(y_test,y_pred))

from tabulate import tabulate
headers = ["name", "score"]
values = sorted(zip(X_train.columns, classifier.feature_importances_), key=lambda x: x[1] * -1)
print(tabulate(values, headers, tablefmt="plain"))
#---------------------------------------------------------------------------------

# PCA
# Learning: PAC accuray is 45%. Model is rubbish. 


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
#cm = confusion_matrix(y_test,y_pred)
#print('Confusion matrix: \n',cm)
print('Classification report: \n',classification_report(y_test,y_pred))
#---------------------------------------------------------------------------------------------------

# LDA
# Learning: LDA accuray is 59%, which is very bad. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.modules[__name__].__dict__.clear()

data = pd.read_csv('train.csv')
X= data.loc[:,(data.columns != 'Cover_Type') ]
y =data.loc[:,'Cover_Type']

# Splitting the dataset into the Training set and Test set
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print('Classification report: \n',classification_report(y_test,y_pred))

#--------------------------------------------------------------------------------------------------------
# Kernal PCA
# Learning: 40% of accuracy. This is worse classifier

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



data = pd.read_csv('train.csv')
X= data.loc[:,(data.columns != 'Cover_Type') ]
y =data.loc[:,'Cover_Type']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print('Classification report: \n',classification_report(y_test,y_pred))


#
## Plot ROC curve
#fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
#plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr, tpr)
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC')
#plt.show()



##################################################################################################################




###############################################################################################################
##############################----------Step 7::Model Data-------------###################################
###############################################################################################################



#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
#    ensemble.AdaBoostClassifier(),
#    ensemble.BaggingClassifier(),
#    ensemble.ExtraTreesClassifier(),
#    ensemble.GradientBoostingClassifier(),
#    ensemble.RandomForestClassifier(),

    #Gaussian Processes
#    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
#    linear_model.PassiveAggressiveClassifier(),
#    linear_model.RidgeClassifierCV(),
#    linear_model.SGDClassifier(),
#    linear_model.Perceptron(),
    
    #Navies Bayes
#    naive_bayes.BernoulliNB(),
#    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
#    neighbors.KNeighborsClassifier(),
#    
#    #SVM
#    svm.SVC(probability=True),
#    svm.NuSVC(probability=True),
#    svm.LinearSVC(),
#    
#    #Trees    
#    tree.DecisionTreeClassifier(),
#    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis()
    
    ]





data = pd.read_csv('train.csv')
X= data.loc[:, (data.columns != 'Cover_Type')
               & (data.columns != 'Id')
             & (data.columns != 'Soil_Type7')
             & (data.columns != 'Soil_Type8')
             & (data.columns != 'Soil_Type9')
             & (data.columns != 'Soil_Type15')
             & (data.columns != 'Soil_Type34')
             & (data.columns != 'Soil_Type36')
             & (data.columns != 'Soil_Type37')
 ]
y =data.loc[:,'Cover_Type']

# Splitting the dataset into the Training set and Test set
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
#note: this is an alternative to train_test_split
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy Min' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = data.loc[:,'Cover_Type']

#index through MLA and save performance to table
row_index = 0

for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, X, y, cv  = cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    MLA_compare.loc[row_index, 'MLA Test Accuracy Min'] = cv_results['test_score'].min()   #let's know the worst that can happen!

    #save MLA predictions - see section 6 for usage
    alg.fit(X, y)
    #MLA_predict[MLA_name] = alg.predict(X)
    
    row_index+=1

    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
#MLA_predict


#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html
sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')

#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')

#MLA_name = alg.__class__.__name__
#MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
#MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
#
##score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
#cv_results = model_selection.cross_validate(alg, X, y, cv  = cv_split)
#
#MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
#MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
#MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
#MLA_compare.loc[row_index, 'MLA Test Accuracy Min'] = cv_results['test_score'].min()   #let's know the worst that can happen!
#
##save MLA predictions - see section 6 for usage
#alg.fit(X, y)
#MLA_predict[MLA_name] = alg.predict(X)



###########################################################

import pandas as pd
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix


data = pd.read_csv('train.csv')
X= data.loc[:, (data.columns != 'Cover_Type')
               & (data.columns != 'Id')
             & (data.columns != 'Soil_Type7')
             & (data.columns != 'Soil_Type8')
             & (data.columns != 'Soil_Type9')
             & (data.columns != 'Soil_Type15')
             & (data.columns != 'Soil_Type34')
             & (data.columns != 'Soil_Type36')
             & (data.columns != 'Soil_Type37')
 ]
y =data.loc[:,'Cover_Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


kfold = KFold(n_splits=10, random_state=0) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]




cv_result = cross_val_score(models[0],X_train,y_train, cv = kfold,scoring = "accuracy")
cv_result=cv_result
xyz.append(cv_result.mean())
std.append(cv_result.std())
accuracy.append(cv_result)






new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
new_models_dataframe2




