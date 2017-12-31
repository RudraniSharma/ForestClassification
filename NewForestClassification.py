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


###############################################################################################################
##############################----------Step 7::Model Data-------------########################################
###############################################################################################################

#Note: 

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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

kfold = KFold(n_splits=10, random_state=0) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=2),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]
#classifiers=['Linear Svm','Radial Svm']#,'Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']
#models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf')]#,LogisticRegression(),KNeighborsClassifier(n_neighbors=2),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]


for i in models:
    model = i
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
print(new_models_dataframe2*100)



new_models_dataframe2['CV Mean'].plot.barh(width=0.8)
plt.title('Average CV Mean Accuracy')
fig=plt.gcf()
fig.set_size_inches(8,5)
plt.show()


plt.subplots(figsize=(12,10))
box=pd.DataFrame(accuracy,index=[classifiers])
box.T.boxplot()

###############################################################################################################
#######################----------End of Step 7::Model Data-------------########################################
###############################################################################################################

