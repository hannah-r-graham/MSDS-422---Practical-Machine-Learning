# %%
### load in data
## explore data 
### create charts with data
#find misisng vals and replace them for cat values
#find missing vals for int values and replace them
# do one hot encoding for both cat and int vals.

# %%
pip install mlxtend


# %%
import math
import pandas as pd
import numpy as np
from operator import itemgetter


import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn import tree
from sklearn.tree import _tree

from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import RandomForestClassifier 

from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.ensemble import GradientBoostingClassifier 

from sklearn.metrics import classification_report, confusion_matrix

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


#dtermines how many rows do display (if none, then no limit)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


#defined targets - target is what we are trying to predict.  
TARGET_F = "TARGET_BAD_FLAG"
TARGET_A = "TARGET_LOSS_AMT"




INFILE = "/Users/hannahgraham/Library/CloudStorage/OneDrive-Personal/Northwestern/msds422_/unit1/Assignment1Data Prep/HMEQ_Loss2.csv"

df = pd.read_csv( INFILE )

#print first few rows of original

print(df.head().T)

# %%
'''Prelim Analysis'''

# %%
#separate out categorical versus numberic data
x = df.describe().T
print(x)

#5960 rows of data in data set. 

# %%
dtype = df.dtypes
print(dtype)

#sep each type into different lists so we can start fixing null values For each item in each list, add how many are null




# %%
#separate out data types into correct lists

objectList = []
integerList = []
floatList = []

for i in dtype.index : 
    if i in ( [ TARGET_F, TARGET_A ] ) : continue
    if dtype[i] in (["object"]) : objectList.append( i )
    if dtype[i] in (["float64"]) : floatList.append( i )
    if dtype[i] in (["int64"]) : integerList.append( i )


print(" OBJECTS ")
print(" ------- ")
for i in objectList :
   print(i)
print("----")
print("\n\n\n")

print(" INTEGER ")
print(" ------- ")
for i in integerList :
   print(i)
print("----")
print("\n\n\n")

print(" FLOAT ")
print(" ----- ")
for i in floatList :
   print(i)
print("\n\n\n")
##
#only categorical data is Reason for loan and what the person's job is. Only integer in data set is Loan amount. The rest are float values in the original data set. 


# %%
#let's take a look at the object data closer: reason and Job. 
#Calculate missing values for each category, then their correlated probability with defaulting and the amount. 


for i in objectList :
   print(" CategoryColumn = ", i )
   g = df.groupby( i )
   print( g[i].count() )
      #finds most common value using mode
   print( "MOST COMMON Value = ", df[i].mode()[0] )  

   #adds up missing value count for that column. SO USEFUL 
   print( "MISSING Values = ", df[i].isna().sum() )
   # print( "\n\n")
   x = g[ TARGET_F ].mean()
   x=round(100*x,1)
   print( "Probability of Default for", x, "%" )
   print( " ................. ")
   x = g[ TARGET_A ].median()
   print( "Amount NOT repayed for", x )
   print(" ===============\n\n\n ")

# %%
#using int and float variables, calculate correlation between variable and whether or not they defaulted and the amount. 

print("\n\n")
print("INTEGER VARIABLES" )
print("\n")
for i in integerList :
   print("Variable=",i )
   g = df.groupby( TARGET_F )
   x = g[ i ].mean()
   print( "Default Probability", x )

   #next line calculates the correlation between the variable and amount defaulted
   c = df[i].corr( df[ TARGET_A ] )
   c = round( 100*c, 1 )
   print( "UNPAID correlation = ", c, "%" )
   print(" ===============\n\n\n ")


## do the same thing with float variables as you did with int variables
##
print("\n\n")
print("FLOAT VARIABLES" )
print("\n")
for i in floatList :
   print("Variable=",i )
   g = df.groupby( TARGET_F )
   x = g[ i ].mean()
   print( "Correlation of defaulting", x )
   c = df[i].corr( df[ TARGET_A ] )
   c = round( 100*c, 1 )
   print( "UNPAID correlation = ", c, "%" )
   print(" ===============\n\n\n ")

# %%
''' Pie Chart Time!'''

#pie chart of break down of jobs
#hist chart of amount defaulted per job of original data, which includs null values.
#

# %%
for i in objectList :
   print(i)
   x = df[ i ].value_counts(dropna=False)
   #print( x )
   theLabels = x.axes[0].tolist()
   print( theLabels )
   theSlices = list(x)
   print( theSlices ) 
   plt.pie( theSlices,
            labels=theLabels,
            startangle = 90,
            shadow=True,
            autopct="%1.0f%%",
            pctdistance=0.85, 
            labeldistance=1.1)
   plt.title("Pie Chart: " + i)
   plt.show()
   print("=====\n\n")

# %%
plt.hist(df [ TARGET_A ] )
plt.xlabel( " Unpaid amounts" )
plt.ylabel( "Occurence/Count" )
plt.show()

# %%

TAcounts = df.groupby('JOB')[TARGET_A].sum()
TAcounts
# plt.hist(TAcounts)
# plt.show()

# %%
plt.hist(df [ TARGET_F ] )
plt.xlabel( " DEFAULTED " )
plt.ylabel( "Occurence/Count" )
plt.show()

# %%
'''Replace Missing Categorical values'''

# %%
# Rewatching videos - reput in lists for safety.

objectList = []
for i in dtype.index:
    if i in ( [ TARGET_F, TARGET_A ] ) : continue
    if dtype[i] in (["object"]) : objectList.append ( i )

# print(" OBJECTS ")
# print(" ------- ")
# for i in objectList :
#    print( i )
# print(" ------- ")
    
for i in objectList : 
    print( i )
    print(df[i].unique() )
    g= df.groupby( i )
    print ( g[i].count() )
    print( "Most common = ", df[i].mode()[0])
    print( " Missing = ", df[i].isna().sum() )



# %%
#For job and reason columns, replace the missing values with "MISSING"in a new column. 
#Keep original column. Will remove below after impution is proved.

for i in objectList :
   if df[i].isna().sum() == 0 : continue
   print(i)
   # print("has missing")
   NAME = "IMP_"+i
   # print( NAME ) 
   df[NAME] = df[i]
   df[NAME] = df[NAME].fillna("MISSING")
   print( "variable",i," has this many null values", df[i].isna().sum() )
   print( "variable",NAME," has this many null values", df[NAME].isna().sum() )
   g = df.groupby( NAME )
   print( g[NAME].count() )
   print( "\n\n")
   #remove original Reason and Job column as we have imputed it
   # df = df.drop( i, axis=1 )



#it worked. Now no null in IMPreason or IMPjob.

# %%
#DISPLAY the amount of missing values for original column job, and how many are missing in IMP_JOB.
for i in ["JOB", "IMP_JOB"]: 
   print( i )
   print( df[i].unique() )
   g = df.groupby( i )
   print( g[i].count() )
   print( " Most Common = ", df[i].mode() [0] )
   print( "Missing = ", df[i].isna().sum() )
   print( "\n\n")

for i in ["REASON", "IMP_REASON"]: 
   print( i )
   print( df[i].unique() )
   g = df.groupby( i )
   print( g[i].count() )
   print( " Most Common = ", df[i].mode() [0] )
   print( "Missing = ", df[i].isna().sum() )
   print( "\n\n")

# %%
#remove original columns Job and Reason as we now have imputed values
df = df.drop(['JOB', 'REASON'], axis=1)
#print snippet of data frame so we can see the columns job and reason have been removed.
print(df.head().T)

# %%
'''Replace missing int values'''

# %%
#sort values into numbers (both float and int)
objList = []
numberList = []

dtype = df.dtypes
for i in dtype.index :
    #print(" here is i .....", i , " ..... and here is the type", dt[i] )
    if i in ( [ TARGET_F, TARGET_A ] ) : continue
    if dtype[i] in (["object"]) : objList.append( i )
    if dtype[i] in (["float64","int64"]) : numberList.append( i )


print(" OBJECTS ")
print(" ------- ")
for i in objList :
   print( i )
print(" ------- ")


print(" NUMBER ")
print(" ------- ")
for i in numberList :
   print( i )
print(" ------- ")

# %%
#replace missing number values with median (not mean - in this case outliers could have a large affect) AND add a flag (M_) to say what was originally missing.

for i in numberList :
   if df[i].isna().sum() == 0 : continue
   #FLAG tells you that the value was originally missing. Uber important to keep
   FLAG = "M_" + i
   IMP = "IMP_" + i
   # print(i)
   # print( FLAG)
   # print( IMP )
   print(" ------- ")
   # add zero to the line: so if true, its converted to 1, if 0, python knows to put 0
   df[ FLAG ] = df[i].isna() + 0
   df[ IMP ] = df[ i ]
   df.loc[ df[IMP].isna(), IMP ] = df[i].median()
   print( "variable",i," has this many missing values",  df[i].isna().sum() )
   print( "variable", FLAG ," has this many missing values",  df[FLAG].isna().sum() )
   print( "variable", IMP ," has this many missing values",  df[IMP].isna().sum() )

   #get rid of original nmerical columns and only keep imputed columns and flagged columns
   df = df.drop( i, axis=1 )
   

#View final columns
print(" ------- ")
print("Columns after replacing numerical missing values and categorical values=" , df.head(3).T)

# %%
'''One hot encoding for cat values'''

# %%
#sort category values and number values into lists again with the updated column names and double check no null values
dtype = df.dtypes

objectList = []
numberList =[]

for i in dtype.index : 
    if i in ( [ TARGET_F, TARGET_A ] ) : continue
    if dtype[i] in (["object"]) : objectList.append( i )
    if dtype[i] in (["float64"]) : numberList.append( i )
    if dtype[i] in (["int64"]) : numberList.append( i )


print(" OBJECTS ")
print(" ------- ")
for i in objectList :
   print(" Class = ", i )
   print( df[i].unique() )
   print("Count of null values", df[i].isna().sum())
print(" ------- ")
print("----")
print("\n\n\n")

print(" NUMBER ")
print(" ------- ")
for i in numberList :
   print(i)
   print("Count of null values", df[i].isna().sum())
print("----")
print("\n\n\n")

#no binary categories. Nor ordinal categories for categorical data. 



# %%
#one hot for categories IMP_Reason and IMP_JOB

for i in objectList :
   #print(" Class = ", i )
   thePrefix = "z_" + i
   #print( thePrefix )
   y = pd.get_dummies( df[i], prefix=thePrefix, dummy_na=False )   
   # next line usees the idea that if you only have 5 options, 
   #you only have 4 one hot encoding because the fifth is implied. Reduces size of one hot encoding. IE if its not yes, its a no.
#    y = pd.get_dummies( df[i], prefix=thePrefix, dummy_na=False , drop_first=True )   
   #print( type(y) )
   #print( y.head().T )
   df = pd.concat( [df, y], axis=1 )
   df = df.drop( i, axis=1 )

print( df.head(5).T )

# %%
#tough stuff is done - now time to the data make it look pretty. 
df=df[sorted(df.columns)]
# print(df.head().T)
#lets put target variables first

TargetColF = df.pop( TARGET_F )
df.insert(0, TARGET_F, TargetColF)
TargetColA = df.pop( TARGET_A )
df.insert(1, TARGET_A, TargetColA )
dt = df.dtypes
print("Final Data types:")
print("-------")
print(dt)
print("-------")
print("\n\n")
print("Stats for final data set")
print("-------")
print(df.describe().T)
print("\n\n")
print("Double check no null values in data set:")
print("-------")
print(df.isna().sum() )


# %%
#print final data set
print("\n\n")
print("Final data set info")
print("-------")
print(df.isna().sum())

# %%
#Done did: EDA, including displays/charts, correlated specific variables to target variables, replaced missing values for both cat and numbers, one hot encoding for all cat variables.  

# %%
"""
SPLIT DATA
"""

# %%


X = df.copy()
X = X.drop( TARGET_F, axis=1 )
X = X.drop( TARGET_A, axis=1 )

Y = df[ [TARGET_F, TARGET_A] ]






# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2 )

print( "FLAG DATA" )
print( "TRAINING = ", X_train.shape )
print( "TEST = ", X_test.shape )



# %%
F = ~ Y_train[ TARGET_A ].isna()
#only the true people of who defaulted will go into w_train
W_train = X_train[F].copy()
Z_train = Y_train[F].copy()

F = ~ Y_test[ TARGET_A ].isna()
W_test = X_test[F].copy()
Z_test = Y_test[F].copy()

# #print( Z_train.describe() )
# #print( Z_test.describe() )
# #print( "\n\n")

# F = Z_train[ TARGET_A ] > 25000
# Z_train.loc[ F, TARGET_A ] = 25000

# F = Z_test[ TARGET_A ] > 25000
# Z_test.loc[ F, [TARGET_A] ] = 25000

# #print( Z_train.describe() )
# #print( Z_test.describe() )
# #print( "\n\n")


# ##print( " ====== ")
# ##
# ##print( "AMOUNT DATA" )
# ##print( "TRAINING = ", W_train.shape )
# ##print( "TEST = ", Z_test.shape )

# %%
# Y_train.head().T
# Z_train.head().T

# %%
"""
DECISION TREE
"""

# %%
# Default PROBABILITY

fm01_Tree = tree.DecisionTreeClassifier( max_depth=5 )
fm01_Tree = fm01_Tree.fit( X_train, Y_train[ TARGET_F ] )


Y_Pred_train = fm01_Tree.predict(X_train)
Y_Pred_test = fm01_Tree.predict(X_test)


print("\n=============\n")
print("DECISION TREE\n")
print("Probability of Default")
print("Accuracy Train:",metrics.accuracy_score(Y_train[TARGET_F], Y_Pred_train))
print("Accuracy Test:",metrics.accuracy_score(Y_test[TARGET_F], Y_Pred_test))
print("\n")

# %%
#weird roc thing

probs = fm01_Tree.predict_proba(X_train)
p1 = probs[:,1]
fpr_train, tpr_train, threshold = metrics.roc_curve( Y_train[TARGET_F], p1)
roc_auc_train = metrics.auc(fpr_train, tpr_train)


# %%
probs = fm01_Tree.predict_proba(X_test)
p1 = probs[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve( Y_test[TARGET_F], p1)
roc_auc_test = metrics.auc(fpr_test, tpr_test)


# %%
fpr_tree = fpr_test
tpr_tree = tpr_test
auc_tree = roc_auc_test

# %%
##
plt.title('TREE ROC CURVE')
plt.plot(fpr_train, tpr_train, label = 'AUC TRAIN = %0.2f' % roc_auc_train, color="blue")
plt.plot(fpr_test, tpr_test, label = 'AUC TEST = %0.2f' % roc_auc_test, color="red")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# %%
# VIEW THE TREEE itself but not great / portable. can copy and paste file into GVEdit

feature_cols = list( X.columns.values )
tree.export_graphviz(fm01_Tree,out_file='tree_f.txt',filled=True, rounded=True, feature_names = feature_cols, impurity=False, class_names=["Good","Bad"]  )



# %%
#tree search and find the variables in the model. 

def getTreeVars( TREE, varNames ) :
   tree_ = TREE.tree_
   varName = [ varNames[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature ]

   nameSet = set()
   for i in tree_.feature :
       if i != _tree.TREE_UNDEFINED :
           nameSet.add( i )
   nameList = list( nameSet )
   parameter_list = list()
   for i in nameList :
       parameter_list.append( varNames[i] )
   return parameter_list


#variables used in decision tree
vars_tree_flag = getTreeVars( fm01_Tree, feature_cols ) 
print(vars_tree_flag)



# %%


# %%
### DAMAGES

# %%
amt_m01_Tree = tree.DecisionTreeRegressor( max_depth= 1 )
amt_m01_Tree = amt_m01_Tree.fit( W_train, Z_train[TARGET_A] )
##
Z_Pred_train = amt_m01_Tree.predict(W_train)
Z_Pred_test = amt_m01_Tree.predict(W_test)
##


RMSE_TRAIN = math.sqrt( metrics.mean_squared_error(Z_train[TARGET_A], Z_Pred_train))
RMSE_TEST = math.sqrt( metrics.mean_squared_error(Z_test[TARGET_A], Z_Pred_test))
##
print("TREE RMSE Train:", RMSE_TRAIN )
print("TREE RMSE Test:", RMSE_TEST )
##
RMSE_TREE = RMSE_TEST
##
feature_cols = list( X.columns.values )
vars_tree_amt = getTreeVars( amt_m01_Tree, feature_cols ) 
tree.export_graphviz(amt_m01_Tree,out_file='tree_a.txt',filled=True, rounded=True, feature_names = feature_cols, impurity=False, precision=0  )

print("\n")
for i in vars_tree_amt :
   print(i)

# %%
#to view of any outliers. Did not remove any in this data set. Maybe that was a mistake. we shall see. But the RMSE train and test comparison is super close so i think its fine to leave it.
print( "MEAN Train", Z_train[TARGET_A].describe() )
print( "MEAN Test", Z_test[TARGET_A].describe() )
print( " ----- \n\n" )

# %%
"""
RANDOM FOREST
"""

# %%
fm01_RF = RandomForestClassifier( n_estimators = 100, random_state=1 )
fm01_RF = fm01_RF.fit( X_train, Y_train[ TARGET_F ] )

Y_Pred_train = fm01_RF.predict(X_train)
Y_Pred_test = fm01_RF.predict(X_test)

# %%
print("\n=============\n")
print("RANDOM FOREST\n")
print("Probability of default")
print("Accuracy Train:",metrics.accuracy_score(Y_train[TARGET_F], Y_Pred_train))
print("Accuracy Test:",metrics.accuracy_score(Y_test[TARGET_F], Y_Pred_test))
print("\n")
##

# %%
#do roc predictor

probs = fm01_RF.predict_proba(X_train)
p1 = probs[:,1]
fpr_train, tpr_train, threshold = metrics.roc_curve( Y_train[TARGET_F], p1)
roc_auc_train = metrics.auc(fpr_train, tpr_train)

probs = fm01_RF.predict_proba(X_test)
p1 = probs[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve( Y_test[TARGET_F], p1)
roc_auc_test = metrics.auc(fpr_test, tpr_test)

# %%
plt.title('RF ROC CURVE')
plt.plot(fpr_train, tpr_train, label = 'AUC TRAIN = %0.2f' % roc_auc_train, color="blue")
plt.plot(fpr_test, tpr_test, label = 'AUC TEST = %0.2f' % roc_auc_test, color="red")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
##

# %%
#save dataset: 
fpr_RF = fpr_test
tpr_RF = tpr_test
auc_RF = roc_auc_test


# %%
#see variables used and how often

def getEnsembleTreeVars( ENSTREE, varNames ) :
   importance = ENSTREE.feature_importances_
   index = np.argsort(importance)
   theList = []
   for i in index :
       imp_val = importance[i]
       if imp_val > np.average( ENSTREE.feature_importances_ ) :
           v = int( imp_val / np.max( ENSTREE.feature_importances_ ) * 100 )
           theList.append( ( varNames[i], v ) )
   theList = sorted(theList,key=itemgetter(1),reverse=True)
   return theList


feature_cols = list( X.columns.values )
vars_RF_flag = getEnsembleTreeVars( fm01_RF, feature_cols )

for i in vars_RF_flag :
   print( i )

# %%
amt_m01_RF = RandomForestRegressor(n_estimators = 100, random_state=1)
amt_m01_RF = amt_m01_RF.fit( W_train, Z_train[TARGET_A] )

Z_Pred_train = amt_m01_RF.predict(W_train)
Z_Pred_test = amt_m01_RF.predict(W_test)

RMSE_TRAIN = math.sqrt( metrics.mean_squared_error(Z_train[TARGET_A], Z_Pred_train))
RMSE_TEST = math.sqrt( metrics.mean_squared_error(Z_test[TARGET_A], Z_Pred_test))

print("RF RMSE Train:", RMSE_TRAIN )
print("RF RMSE Test:", RMSE_TEST )

RMSE_RF = RMSE_TEST

feature_cols = list( X.columns.values )
vars_RF_amt = getEnsembleTreeVars( amt_m01_RF, feature_cols )

for i in vars_RF_amt :
   print( i )
##
##

# %%
"""
GRADIENT BOOSTING
"""

# %%
fm01_GB = GradientBoostingClassifier( random_state=1 )
fm01_GB = fm01_GB.fit( X_train, Y_train[ TARGET_F ] )

Y_Pred_train = fm01_GB.predict(X_train)
Y_Pred_test = fm01_GB.predict(X_test)
##

# %%
print("\n=============\n")
print("GRADIENT BOOSTING\n")
print("Probability of default")
print("Accuracy Train:",metrics.accuracy_score(Y_train[TARGET_F], Y_Pred_train))
print("Accuracy Test:",metrics.accuracy_score(Y_test[TARGET_F], Y_Pred_test))
print("\n")


# %%
probs = fm01_GB.predict_proba(X_train)
p1 = probs[:,1]
fpr_train, tpr_train, threshold = metrics.roc_curve( Y_train[TARGET_F], p1)
roc_auc_train = metrics.auc(fpr_train, tpr_train)

probs = fm01_GB.predict_proba(X_test)
p1 = probs[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve( Y_test[TARGET_F], p1)
roc_auc_test = metrics.auc(fpr_test, tpr_test)

fpr_GB = fpr_test
tpr_GB = tpr_test
auc_GB = roc_auc_test


feature_cols = list( X.columns.values )
vars_GB_flag = getEnsembleTreeVars( fm01_GB, feature_cols )


for i in vars_GB_flag :
   print(i)
##

# %%
plt.title('GB ROC CURVE')
plt.plot(fpr_train, tpr_train, label = 'AUC TRAIN = %0.2f' % roc_auc_train, color="blue")
plt.plot(fpr_test, tpr_test, label = 'AUC TEST = %0.2f' % roc_auc_test, color="red")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
##
##

# %%
# predicting damages of the car

# %%

amt_m01_GB = GradientBoostingRegressor(random_state=1)
amt_m01_GB = amt_m01_GB.fit( W_train, Z_train[TARGET_A] )

Z_Pred_train = amt_m01_GB.predict(W_train)
Z_Pred_test = amt_m01_GB.predict(W_test)

RMSE_TRAIN = math.sqrt( metrics.mean_squared_error(Z_train[TARGET_A], Z_Pred_train))
RMSE_TEST = math.sqrt( metrics.mean_squared_error(Z_test[TARGET_A], Z_Pred_test))

print("GB RMSE Train:", RMSE_TRAIN )
print("GB RMSE Test:", RMSE_TEST )

RMSE_GB = RMSE_TEST

feature_cols = list( X.columns.values )
vars_GB_amt = getEnsembleTreeVars( amt_m01_GB, feature_cols )


for i in vars_GB_amt :
   print(i)


# %%
plt.title('MODELS ROC CURVE')
plt.plot(fpr_tree, tpr_tree, label = 'AUC TREE = %0.2f' % auc_tree, color="red")
plt.plot(fpr_RF, tpr_RF, label = 'AUC RF = %0.2f' % auc_RF, color="green")
plt.plot(fpr_GB, tpr_GB, label = 'AUC GB = %0.2f' % auc_GB, color="blue")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



print("Root Mean Square Average For Damages")
print("TREE", RMSE_TREE)
print("RF", RMSE_RF)
print("GB", RMSE_GB)

# %%
''' now we need to handle outliers'''

# %%
# split by data type and print out variables 
dt = df.dtypes
numList = []
for i in dt.index :
    #print(i, dt[i])
    if i in ( [ TARGET_F, TARGET_A ] ) : continue
    if dt[i] in (["float64","int64"]) : numList.append( i )

for i in numList:
    print(i)


# %%
print(" number ")
print(" ----- ")
for i in numList: 
    print( df[i]. describe() )
    print( " -------\n ")

# %%
for i in numList :
    print( i )
    plt.hist ( df[ i ])
    plt.xlabel ( i )
    plt.show()

# %%
#never let the value exceed 3 x the StD
for i in numList :
    theMean = df[i].mean()
    theSD = df[i].std()
    theMax = df[i].max()
    theCutoff = round( theMean + 3*theSD )
    if theMax < theCutoff : continue
    #flag if you fixed an outlier
    FLAG = "O_" + i
    TRUNC = "TRUNC_" + i
    df[ FLAG ] = ( df[i] > theCutoff )+ 0
    df[ TRUNC ] = df[ i ]
    df.loc[ df[TRUNC] > theCutoff, TRUNC ] = theCutoff
    df = df.drop( i, axis=1 )

# %%
#check out vars after we fix them
dt = df.dtypes
numList = []
for i in dt.index :
    # print(i, dt[i])
    if i in ( [ TARGET_F, TARGET_A ] ) : continue
    if dt[i] in (["float64","int64"]) : numList.append( i )

for i in numList:
    print(i)

# %%
print(" number ")
print(" ----- ")
for i in numList: 
    print( df[i]. describe() )
    print( " -------\n ")

# %%
for i in numList :
    print( i )
    plt.hist ( df[ i ])
    plt.xlabel ( i )
    plt.show()

# %%
"""
MODEL ACCURACY METRICS (FUNCTIONS)
"""

def getProbAccuracyScores( NAME, MODEL, X, Y ) :
    pred = MODEL.predict( X )
    probs = MODEL.predict_proba( X )
    acc_score = metrics.accuracy_score(Y, pred)
    p1 = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve( Y, p1)
    auc = metrics.auc(fpr,tpr)
    return [NAME, acc_score, fpr, tpr, auc]

# %%
def print_ROC_Curve( TITLE, LIST ) :
    fig = plt.figure(figsize=(6,4))
    plt.title( TITLE )
    for theResults in LIST :
        NAME = theResults[0]
        fpr = theResults[2]
        tpr = theResults[3]
        auc = theResults[4]
        theLabel = "AUC " + NAME + ' %0.2f' % auc
        plt.plot(fpr, tpr, label = theLabel )
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# %%
def print_Accuracy( TITLE, LIST ) :
    print( TITLE )
    print( "======" )
    for theResults in LIST :
        NAME = theResults[0]
        ACC = theResults[1]
        print( NAME, " = ", ACC )
    print( "------\n\n" )

# %%
def getAmtAccuracyScores( NAME, MODEL, X, Y ) :
    pred = MODEL.predict( X )
    MEAN = Y.mean()
    RMSE = math.sqrt( metrics.mean_squared_error( Y, pred))
    return [NAME, RMSE, MEAN]

# %%
'''TREE BASED'''

# %%
#crash probability tree based
WHO = "TREE"

CLM = tree.DecisionTreeClassifier( max_depth=4 )
CLM = CLM.fit( X_train, Y_train[ TARGET_F ] )

TRAIN_CLM = getProbAccuracyScores( WHO + "_Train", CLM, X_train, Y_train[ TARGET_F ] )
TEST_CLM = getProbAccuracyScores( WHO, CLM, X_test, Y_test[ TARGET_F ] )

print_ROC_Curve( WHO, [ TRAIN_CLM, TEST_CLM ] ) 
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )

feature_cols = list( X.columns.values )
tree.export_graphviz(CLM,out_file='tree_f.txt',filled=True, rounded=True, feature_names = feature_cols, impurity=False, class_names=["Good","Bad"]  )
vars_tree_flag = getTreeVars( CLM, feature_cols ) 

# %%
print_Accuracy( " CLASSIFICATION ACCURACY ", [TRAIN_CLM, TEST_CLM])

# %%

feature_cols = list( X.columns.values )
tree.export_graphviz(CLM,out_file='NEWtree_f.txt',filled=True, rounded=True, feature_names = feature_cols, impurity=False, class_names=["Good","Bad"]  )
vars_tree_flag = getTreeVars( CLM, feature_cols)

#what variables did the tree like? 

vars_tree_flag

# %%
''' Damages'''

# %%
AMT = tree.DecisionTreeRegressor( max_depth= 4 )
AMT = AMT.fit( W_train, Z_train[TARGET_A] )

TRAIN_AMT = getAmtAccuracyScores( WHO + "_Train", AMT, W_train, Z_train[TARGET_A] )
TEST_AMT = getAmtAccuracyScores( WHO, AMT, W_test, Z_test[TARGET_A] )
#print_Accuracy( WHO + " RMSE ACCURACY", [ TRAIN_AMT, TEST_AMT ] )

feature_cols = list( X.columns.values )
vars_tree_amt = getTreeVars( AMT, feature_cols ) 
tree.export_graphviz(AMT,out_file='tree_a.txt',filled=True, rounded=True, feature_names = feature_cols, impurity=False, precision=0  )


TREE_CLM = TEST_CLM.copy()
TREE_AMT = TEST_AMT.copy()

# %%
'''Random Forest: CRASH PRED'''

# %%
def getEnsembleTreeVars( ENSTREE, varNames ) :
    importance = ENSTREE.feature_importances_
    index = np.argsort(importance)
    theList = []
    for i in index :
        imp_val = importance[i]
        if imp_val > np.average( ENSTREE.feature_importances_ ) :
            v = int( imp_val / np.max( ENSTREE.feature_importances_ ) * 100 )
            theList.append( ( varNames[i], v ) )
    theList = sorted(theList,key=itemgetter(1),reverse=True)
    return theList

# %%
WHO = "RF"

CLM = RandomForestClassifier( n_estimators = 25, random_state=1 )
CLM = CLM.fit( X_train, Y_train[ TARGET_F ] )

TRAIN_CLM = getProbAccuracyScores( WHO + "_Train", CLM, X_train, Y_train[ TARGET_F ] )
TEST_CLM = getProbAccuracyScores( WHO, CLM, X_test, Y_test[ TARGET_F ] )

print_ROC_Curve( WHO, [ TRAIN_CLM, TEST_CLM ] ) 
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )


feature_cols = list( X.columns.values )
vars_RF_flag = getEnsembleTreeVars( CLM, feature_cols )

# %%
'''DAMAGES FOR RANDOM FOREST'''

# %%
AMT = RandomForestRegressor(n_estimators = 100, random_state=1)
AMT = AMT.fit( W_train, Z_train[TARGET_A] )

TRAIN_AMT = getAmtAccuracyScores( WHO + "_Train", AMT, W_train, Z_train[TARGET_A] )
TEST_AMT = getAmtAccuracyScores( WHO, AMT, W_test, Z_test[TARGET_A] )
print_Accuracy( WHO + " RMSE ACCURACY", [ TRAIN_AMT, TEST_AMT ] )

feature_cols = list( X.columns.values )
vars_RF_amt = getEnsembleTreeVars( AMT, feature_cols )

for i in vars_RF_amt :
   print( i )

RF_CLM = TEST_CLM.copy()
RF_AMT = TEST_AMT.copy()


# %%
'''Gradient boosting'''

# %%
WHO = "GB"

CLM = GradientBoostingClassifier( random_state=1 )
CLM = CLM.fit( X_train, Y_train[ TARGET_F ] )

TRAIN_CLM = getProbAccuracyScores( WHO + "_Train", CLM, X_train, Y_train[ TARGET_F ] )
TEST_CLM = getProbAccuracyScores( WHO, CLM, X_test, Y_test[ TARGET_F ] )

print_ROC_Curve( WHO, [ TRAIN_CLM, TEST_CLM ] ) 
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )


feature_cols = list( X.columns.values )
vars_GB_flag = getEnsembleTreeVars( CLM, feature_cols )


# %%
'''Damages for GB'''

# %%

AMT = GradientBoostingRegressor(random_state=1)
AMT = AMT.fit( W_train, Z_train[TARGET_A] )

TRAIN_AMT = getAmtAccuracyScores( WHO + "_Train", AMT, W_train, Z_train[TARGET_A] )
TEST_AMT = getAmtAccuracyScores( WHO, AMT, W_test, Z_test[TARGET_A] )
print_Accuracy( WHO + " RMSE ACCURACY", [ TRAIN_AMT, TEST_AMT ] )

feature_cols = list( X.columns.values )
vars_GB_amt = getEnsembleTreeVars( AMT, feature_cols )

for i in vars_GB_amt :
   print( i )

GB_CLM = TEST_CLM.copy()
GB_AMT = TEST_AMT.copy()


# %% [markdown]
# Linear regression/log regression
# 
# 
# How do you decide which variables to use in your model? 
# - tree based model
# - forward stepwise (start with a model with no variable, then add variables over time until the model stops improving)
# - backward stepwise (reverse of forward, start with all variables then remove as you go)
# - 

# %%
'''' REGRESSION ALL VARIABLES'''

# %%

def getCoefLogit( MODEL, TRAIN_DATA ) :
    varNames = list( TRAIN_DATA.columns.values )
    coef_dict = {}
    coef_dict["INTERCEPT"] = MODEL.intercept_[0]
    for coef, feat in zip(MODEL.coef_[0],varNames):
        coef_dict[feat] = coef
    print("\nCRASH")
    print("---------")
    print("Total Variables: ", len( coef_dict ) )
    for i in coef_dict :
        print( i, " = ", coef_dict[i]  )

# %%
WHO = "REG_ALL"

CLM = LogisticRegression( solver='newton-cg', max_iter=1000 )
CLM = CLM.fit( X_train, Y_train[ TARGET_F ] )

TRAIN_CLM = getProbAccuracyScores( WHO + "_Train", CLM, X_train, Y_train[ TARGET_F ] )
TEST_CLM = getProbAccuracyScores( WHO, CLM, X_test, Y_test[ TARGET_F ] )

print_Accuracy( WHO + " RMSE ACCURACY", [ TRAIN_CLM, TEST_CLM ] )

print_ROC_Curve( WHO, [ TRAIN_CLM, TEST_CLM ] ) 
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )

# %%
'''REGRESSION ALL VARIABLES: DAMAGES'''

# %%

def getCoefLinear( MODEL, TRAIN_DATA ) :
    varNames = list( TRAIN_DATA.columns.values )
    coef_dict = {}
    coef_dict["INTERCEPT"] = MODEL.intercept_
    for coef, feat in zip(MODEL.coef_,varNames):
        coef_dict[feat] = coef
    print("\nDAMAGES")
    print("---------")
    print("Total Variables: ", len( coef_dict ) )
    for i in coef_dict :
        print( i, " = ", coef_dict[i]  )

# %%
AMT = LinearRegression()
AMT = AMT.fit( W_train, Z_train[TARGET_A] )

TRAIN_AMT = getAmtAccuracyScores( WHO + "_Train", AMT, W_train, Z_train[TARGET_A] )
TEST_AMT = getAmtAccuracyScores( WHO, AMT, W_test, Z_test[TARGET_A] )
print_Accuracy( WHO + " RMSE ACCURACY", [ TRAIN_AMT, TEST_AMT ] )


varNames = list( X_train.columns.values )

REG_ALL_CLM_COEF = getCoefLogit( CLM, X_train )
REG_ALL_AMT_COEF = getCoefLinear( AMT, X_train )

REG_ALL_CLM = TEST_CLM.copy()
REG_ALL_AMT = TEST_AMT.copy()


# %%

"""
REGRESSION DECISION TREE
"""

# %%
WHO = "REG_TREE"

CLM = LogisticRegression( solver='newton-cg', max_iter=1000 )
CLM = CLM.fit( X_train[vars_tree_flag], Y_train[ TARGET_F ] )

TRAIN_CLM = getProbAccuracyScores( WHO + "_Train", CLM, X_train[vars_tree_flag], Y_train[ TARGET_F ] )
TEST_CLM = getProbAccuracyScores( WHO, CLM, X_test[vars_tree_flag], Y_test[ TARGET_F ] )

print_ROC_Curve( WHO, [ TRAIN_CLM, TEST_CLM ] ) 
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )

# %%
AMT = LinearRegression()
AMT = AMT.fit( W_train[vars_tree_amt], Z_train[TARGET_A] )

TRAIN_AMT = getAmtAccuracyScores( WHO + "_Train", AMT, W_train[vars_tree_amt], Z_train[TARGET_A] )
TEST_AMT = getAmtAccuracyScores( WHO, AMT, W_test[vars_tree_amt], Z_test[TARGET_A] )
print_Accuracy( WHO + " RMSE ACCURACY", [ TRAIN_AMT, TEST_AMT ] )


varNames = list( X_train.columns.values )

#REG_TREE_CLM_COEF = getCoefLogit( CLM, X_train[vars_tree_flag] )
#REG_TREE_AMT_COEF = getCoefLinear( AMT, X_train[vars_tree_amt] )

REG_TREE_CLM = TEST_CLM.copy()
REG_TREE_AMT = TEST_AMT.copy()

# %%
"""
REGRESSION RANDOM FOREST
"""

# %%
WHO = "REG_RF"


print("\n\n")
RF_flag = []
for i in vars_RF_flag :
    print(i)
    theVar = i[0]
    RF_flag.append( theVar )

print("\n\n")
RF_amt = []
for i in vars_RF_amt :
    print(i)
    theVar = i[0]
    RF_amt.append( theVar )


CLM = LogisticRegression( solver='newton-cg', max_iter=1000 )
CLM = CLM.fit( X_train[RF_flag], Y_train[ TARGET_F ] )

TRAIN_CLM = getProbAccuracyScores( WHO + "_Train", CLM, X_train[RF_flag], Y_train[ TARGET_F ] )
TEST_CLM = getProbAccuracyScores( WHO, CLM, X_test[RF_flag], Y_test[ TARGET_F ] )

print_ROC_Curve( WHO, [ TRAIN_CLM, TEST_CLM ] ) 
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )

# %%
# Regression Random forest DAMAGES

AMT = LinearRegression()
AMT = AMT.fit( W_train[RF_amt], Z_train[TARGET_A] )

TRAIN_AMT = getAmtAccuracyScores( WHO + "_Train", AMT, W_train[RF_amt], Z_train[TARGET_A] )
TEST_AMT = getAmtAccuracyScores( WHO, AMT, W_test[RF_amt], Z_test[TARGET_A] )
print_Accuracy( WHO + " RMSE ACCURACY", [ TRAIN_AMT, TEST_AMT ] )


REG_RF_CLM_COEF = getCoefLogit( CLM, X_train[RF_flag] )
REG_RF_AMT_COEF = getCoefLinear( AMT, X_train[RF_amt] )

REG_RF_CLM = TEST_CLM.copy()
REG_RF_AMT = TEST_AMT.copy()


# %%
"""
REGRESSION GRADIENT BOOSTING
"""

# %%
WHO = "REG_GB"


print("\n\n")
GB_flag = []
for i in vars_GB_flag :
    print(i)
    theVar = i[0]
    GB_flag.append( theVar )

print("\n\n")
GB_amt = []
for i in vars_GB_amt :
    print(i)
    theVar = i[0]
    GB_amt.append( theVar )


CLM = LogisticRegression( solver='newton-cg', max_iter=1000 )
CLM = CLM.fit( X_train[GB_flag], Y_train[ TARGET_F ] )

TRAIN_CLM = getProbAccuracyScores( WHO + "_Train", CLM, X_train[GB_flag], Y_train[ TARGET_F ] )
TEST_CLM = getProbAccuracyScores( WHO, CLM, X_test[GB_flag], Y_test[ TARGET_F ] )

print_ROC_Curve( WHO, [ TRAIN_CLM, TEST_CLM ] ) 
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )


# %%
# DAMAGES

AMT = LinearRegression()
AMT = AMT.fit( W_train[GB_amt], Z_train[TARGET_A] )

TRAIN_AMT = getAmtAccuracyScores( WHO + "_Train", AMT, W_train[GB_amt], Z_train[TARGET_A] )
TEST_AMT = getAmtAccuracyScores( WHO, AMT, W_test[GB_amt], Z_test[TARGET_A] )
print_Accuracy( WHO + " RMSE ACCURACY", [ TRAIN_AMT, TEST_AMT ] )

REG_GB_CLM_COEF = getCoefLogit( CLM, X_train[GB_flag] )
REG_GB_AMT_COEF = getCoefLinear( AMT, X_train[GB_amt] )

REG_GB_CLM = TEST_CLM.copy()
REG_GB_AMT = TEST_AMT.copy()

# %%
'''Step wise'''

# %%
U_train = X_train[ vars_tree_flag ]
stepVarNames = list( U_train.columns.values )
maxCols = U_train.shape[1]

sfs = SFS( LogisticRegression( solver='newton-cg', max_iter=100 ),
           k_features=( 1, maxCols ),
           forward=True,
           floating=False,
           cv=3
           )
sfs.fit(U_train.values, Y_train[ TARGET_F ].values)

theFigure = plot_sfs(sfs.get_metric_dict(), kind=None )
plt.title('Default PROBABILITY Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()

dfm = pd.DataFrame.from_dict( sfs.get_metric_dict()).T
dfm = dfm[ ['feature_names', 'avg_score'] ]
dfm.avg_score = dfm.avg_score.astype(float)

print(" ................... ")
maxIndex = dfm.avg_score.argmax()
print("argmax")
print( dfm.iloc[ maxIndex, ] )
print(" ................... ")

stepVars = dfm.iloc[ maxIndex, ]
stepVars = stepVars.feature_names
print( stepVars )

finalStepVars = []
for i in stepVars :
    index = int(i)
    try :
        theName = stepVarNames[ index ]
        finalStepVars.append( theName )
    except :
        pass

for i in finalStepVars :
    print(i)

U_train = X_train[ finalStepVars ]
U_test = X_test[ finalStepVars ]



V_train = W_train[ GB_amt ]
stepVarNames = list( V_train.columns.values )
maxCols = V_train.shape[1]

sfs = SFS( LinearRegression(),
           k_features=( 1, maxCols ),
           forward=True,
           floating=False,
           scoring = 'r2',
           cv=5
           )
sfs.fit(V_train.values, Z_train[ TARGET_A ].values)

theFigure = plot_sfs(sfs.get_metric_dict(), kind=None )
plt.title('DAMAGES Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()

dfm = pd.DataFrame.from_dict( sfs.get_metric_dict()).T
dfm = dfm[ ['feature_names', 'avg_score'] ]
dfm.avg_score = dfm.avg_score.astype(float)

print(" ................... ")
maxIndex = dfm.avg_score.argmax()
print("argmax")
print( dfm.iloc[ maxIndex, ] )
print(" ................... ")

stepVars = dfm.iloc[ maxIndex, ]
stepVars = stepVars.feature_names
print( stepVars )

finalStepVars = []
for i in stepVars :
    index = int(i)
    try :
        theName = stepVarNames[ index ]
        finalStepVars.append( theName )
    except :
        pass

for i in finalStepVars :
    print(i)

V_train = W_train[ finalStepVars ]
V_test = W_test[ finalStepVars ]

# %%
"""
REGRESSION 
"""

WHO = "REG_STEPWISE"

CLM = LogisticRegression( solver='newton-cg', max_iter=1000 )
CLM = CLM.fit( U_train, Y_train[ TARGET_F ] )

TRAIN_CLM = getProbAccuracyScores( WHO + "_Train", CLM, U_train, Y_train[ TARGET_F ] )
TEST_CLM = getProbAccuracyScores( WHO, CLM, U_test, Y_test[ TARGET_F ] )

print_ROC_Curve( WHO, [ TRAIN_CLM, TEST_CLM ] ) 
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )


# DAMAGES

AMT = LinearRegression()
AMT = AMT.fit( V_train, Z_train[TARGET_A] )

TRAIN_AMT = getAmtAccuracyScores( WHO + "_Train", AMT, V_train, Z_train[TARGET_A] )
TEST_AMT = getAmtAccuracyScores( WHO, AMT, V_test, Z_test[TARGET_A] )
print_Accuracy( WHO + " RMSE ACCURACY", [ TRAIN_AMT, TEST_AMT ] )

REG_STEP_CLM_COEF = getCoefLogit( CLM, U_train )
REG_STEP_AMT_COEF = getCoefLinear( AMT, V_train )

REG_STEP_CLM = TEST_CLM.copy()
REG_STEP_AMT = TEST_AMT.copy()








# %%
ALL_CLM = [ TREE_CLM, RF_CLM, GB_CLM, REG_ALL_CLM, REG_TREE_CLM, REG_RF_CLM, REG_GB_CLM, REG_STEP_CLM ]

ALL_CLM = sorted( ALL_CLM, key = lambda x: x[4], reverse=True )
print_ROC_Curve( WHO, ALL_CLM ) 

ALL_CLM = sorted( ALL_CLM, key = lambda x: x[1], reverse=True )
print_Accuracy( "ALL CLASSIFICATION ACCURACY", ALL_CLM )


ALL_AMT = [ TREE_AMT, RF_AMT, GB_AMT, REG_ALL_AMT, REG_TREE_AMT, REG_RF_AMT, REG_GB_AMT, REG_STEP_AMT ]
ALL_AMT = sorted( ALL_AMT, key = lambda x: x[1] )
print_Accuracy( "ALL DAMAGE MODEL ACCURACY", ALL_AMT )

# %%
pip install tensorflow

# %%
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

# %%
'''tensor flow flow model to predict loan defaults'''

# %%
theScaler = MinMaxScaler()
theScaler.fit( X_train )

# %%
def get_TF_ProbAccuracyScores( NAME, MODEL, X, Y ) :
    probs = MODEL.predict( X )
    pred_list = []
    for p in probs :
        pred_list.append( np.argmax( p ) )
    pred = np.array( pred_list )
    acc_score = metrics.accuracy_score(Y, pred)
    p1 = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve( Y, p1)
    auc = metrics.auc(fpr,tpr)
    return [NAME, acc_score, fpr, tpr, auc]





# %%
WHO = "Tensor_FLow"

U_train = theScaler.transform( X_train )
U_test = theScaler.transform( X_test )

U_train = pd.DataFrame( U_train )
U_test = pd.DataFrame( U_test )

U_train.columns = list( X_train.columns.values )
U_test.columns = list( X_train.columns.values )

#comment the below out if you want to use all variables
U_train = U_train[ GB_flag ]
U_test = U_test[ GB_flag ]

# %%
F_theShapeSize = U_train.shape[1]
F_theActivation = tf.keras.activations.softplus
F_theLossMetric = tf.keras.losses.SparseCategoricalCrossentropy()
F_theOptimizer = tf.keras.optimizers.Adam()
F_theEpochs = 100

F_theUnits = int( 2*F_theShapeSize / 3 )

F_LAYER_01 = tf.keras.layers.Dense( units=F_theUnits, activation=F_theActivation, input_dim=F_theShapeSize )
F_LAYER_DROP = tf.keras.layers.Dropout( 0.2 )
F_LAYER_02 = tf.keras.layers.Dense( units=F_theUnits, activation=F_theActivation )
F_LAYER_OUTPUT = tf.keras.layers.Dense( units=2, activation=tf.keras.activations.softmax )



# %%
CLM = tf.keras.Sequential()
CLM.add( F_LAYER_01 )
CLM.add( F_LAYER_DROP )
CLM.add( F_LAYER_02 )
CLM.add( F_LAYER_OUTPUT )
CLM.compile( loss=F_theLossMetric, optimizer=F_theOptimizer)
CLM.fit( U_train, Y_train[TARGET_F], epochs=F_theEpochs, verbose=False )

TRAIN_CLM = get_TF_ProbAccuracyScores( WHO + "_Train", CLM, U_train, Y_train[ TARGET_F ] )
TEST_CLM = get_TF_ProbAccuracyScores( WHO, CLM, U_test, Y_test[ TARGET_F ] )

print_ROC_Curve( WHO, [ TRAIN_CLM, TEST_CLM ] ) 
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )

# %%
'''Tensor flow model to predict loss given default'''

# %%
V_train = theScaler.transform( W_train )
V_test = theScaler.transform( W_test )

V_train = pd.DataFrame( V_train )
V_test = pd.DataFrame( V_test )

V_train.columns = list( W_train.columns.values )
V_test.columns = list( W_train.columns.values )

#comment the below out if you want to use all variables
V_train = V_train[ GB_amt ]
V_test = V_test[ GB_amt ]


# %%
A_theShapeSize = V_train.shape[1]
A_theActivation = tf.keras.activations.softplus
A_theLossMetric = tf.keras.losses.MeanSquaredError()
A_theOptimizer = tf.keras.optimizers.Adam()
A_theEpochs = 800

A_theUnits = int( 2*A_theShapeSize  )

A_LAYER_01 = tf.keras.layers.Dense( units=A_theUnits, activation=A_theActivation, input_dim=A_theShapeSize )
A_LAYER_DROP = tf.keras.layers.Dropout( 0.2 )
A_LAYER_02 = tf.keras.layers.Dense( units=A_theUnits, activation=A_theActivation )
A_LAYER_OUTPUT = tf.keras.layers.Dense( units=1, activation=tf.keras.activations.linear )

AMT = tf.keras.Sequential()
AMT.add( A_LAYER_01 )
AMT.add( A_LAYER_DROP )
# AMT.add( A_LAYER_02 )
AMT.add( A_LAYER_OUTPUT )
AMT.compile( loss=A_theLossMetric, optimizer=A_theOptimizer)
AMT.fit( V_train, Z_train[TARGET_A], epochs=A_theEpochs, verbose=False )


TRAIN_AMT = getAmtAccuracyScores( WHO + "_Train", AMT, V_train[GB_amt], Z_train[TARGET_A] )
TEST_AMT = getAmtAccuracyScores( WHO, AMT, V_test[GB_amt], Z_test[TARGET_A] )
print_Accuracy( WHO + " RMSE ACCURACY", [ TRAIN_AMT, TEST_AMT ] )

TF_CLM = TEST_CLM.copy()
TF_AMT = TEST_AMT.copy()

# %%
ALL_CLM = [ TREE_CLM, RF_CLM, GB_CLM, REG_ALL_CLM, REG_TREE_CLM, REG_RF_CLM, REG_GB_CLM, TF_CLM ]

ALL_CLM = sorted( ALL_CLM, key = lambda x: x[4], reverse=True )
print_ROC_Curve( WHO, ALL_CLM ) 

ALL_CLM = sorted( ALL_CLM, key = lambda x: x[1], reverse=True )
print_Accuracy( "ALL CLASSIFICATION ACCURACY", ALL_CLM )



ALL_AMT = [ TREE_AMT, RF_AMT, GB_AMT, REG_ALL_AMT, REG_TREE_AMT, REG_RF_AMT, REG_GB_AMT, TF_AMT ]
ALL_AMT = sorted( ALL_AMT, key = lambda x: x[1] )
print_Accuracy( "ALL DAMAGE MODEL ACCURACY", ALL_AMT )



