import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def readCSV(iFile):
    print("Reading in File: ", iFile)
    df=pd.read_csv(iFile, header=0, delimiter=",")
    print("Column Data Types:")
    for label in df.columns:
        print(df[label].dtype," ", end = '')
    print("\nColumns: ")
    for label in df.columns:
        print(label," ", end = '')
    print("\n")
    return df

def corrMatrix(df,target): #step 3
    dfCorr=df.corr()
    print(dfCorr)
    print(dfCorr[target].loc[dfCorr[target]>.5])
    return dfCorr

def corrGraphs(df): #scatter and line
    for label in df.columns:
        if df[label].dtype in ['int64', 'float64']:
            df.plot(kind='scatter', x=)
            plt.show()
        elif df[label].dtype == 'object':
            replaceWith = "UNKNOWN"
        newColumn = df[label].replace(to_replace=None, value=replaceWith)
        df.assign(label=newColumn)

def fillnan(df):
    for label in df.columns:
        if df[label].hasnans:
            print(label)
            print(df[label].dtype)
            if df[label].dtype in ['int64','float64']:
                replaceWith=df[label].mean()
                print("mean", replaceWith)
            elif df[label].dtype=='object':
                 replaceWith="UNKNOWN"
            newColumn=df[label].replace(to_replace=None, value=replaceWith)
            df.assign(label=newColumn)
    return df

housing=readCSV("/Users/hannaalbright1/Desktop/CSCI 183/housing.csv")
houseData=readCSV("/Users/hannaalbright1/Desktop/CSCI 183/kc_house_data.csv")
#housing=fillnan(housing)
#print("i")
houseCorr=corrMatrix(housing, "median_house_value")
corrGraphs(houseCorr)


# 3. Find the correlation matrix for this dataset. Report which features tend to have a
# high correlation with the target variable.


# 4. Create and compile as many graphs (feature vs target variable) as you can using
# the matplotlib library [https://matplotlib.org/gallery/index.html] for the given
# dataset.
# 5. Based on the graphs in step 4, identify features that have a linear relationship
# with the target variable.
# 6. Selecting different features from step 5, implement a linear regression algorithm
# and find the slope, the intercept and the error of the regression model.