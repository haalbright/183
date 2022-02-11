import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
PATH="/Users/hannaalbright1/Desktop/CSCI 183/"

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

# 3. Find the correlation matrix for this dataset. Report which features tend to have a
# high correlation with the target variable.
def corrMatrix(df,target): #step 3
    print("High correlation values with", target)
    dfCorr=df.corr()
    targetCorr=dfCorr[target].drop(target)
    for label, val in targetCorr[targetCorr>=.5].items():
        print(label," has a correlation value of", val)
    return dfCorr

# 4. Create and compile as many graphs (feature vs target variable) as you can using
# the matplotlib library [https://matplotlib.org/gallery/index.html] for the given
# dataset.
def plotGraphs(df, target): #scatter and line
    for label in df.drop(target, axis=1).columns:
        if df[label].dtype in ['int64', 'float64']:
            df.plot.scatter(y=target, x=label)
            name = PATH+target+"/"+label + "graph.png"
            plt.savefig(name)
            plt.close()


# 6. Selecting different features from step 5, implement a linear regression algorithm
# and find the slope, the intercept and the error of the regression model.
def linearReg(df,target,features):
    yVal = df[target].to_numpy().reshape(-1,1)
    for label in df[features]:
        print("\n",label)
        xVal=df[label].to_numpy().reshape(-1,1)
        lReg=LinearRegression()
        lReg.fit(xVal,yVal)
        intercept= lReg.intercept_
        slope = lReg.coef_[0]
        predictedY = lReg.predict(xVal)
        print("Slope: ", slope, " Intercept: ", intercept, "Mean Squared Error", mean_squared_error(y_true=df[target], y_pred=predictedY))
        plt.scatter(xVal,yVal)
        plt.plot(xVal, predictedY, color="red")
        name = PATH+ "/linReg"+target +"/"+ label + "linearRegGraph.png"
        plt.savefig(name)
        plt.close()


housing=readCSV(PATH+"housing.csv")
houseCorr=corrMatrix(housing, "median_house_value")
houseCorr.to_csv(PATH+"correlationHousing.csv")
plotGraphs(housing,"median_house_value")
linearReg(housing,"median_house_value",['median_income'])

houseData=readCSV(PATH+"kc_house_data.csv")
houseDataCorr=corrMatrix(houseData, "price")
houseDataCorr.to_csv(PATH+"correlationKC_house_data.csv")
plotGraphs(houseData,"price")
linearReg(houseData,"price",['sqft_living', 'sqft_living15', 'sqft_above', 'grade', 'bathrooms'])

# 5. Based on the graphs in step 4, identify features that have a linear relationship
# with the target variable.
#linear relationships with price in kc_house_data.csv: sqft_living, sqft_living15, sqft_above, grade, and bathrooms
#linear relationships with median_house_value in housing.csv: median income (although it isn't a strong linear relationship)


