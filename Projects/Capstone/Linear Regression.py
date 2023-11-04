import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv('//home/derrick/diamonds.csv') #Read in the .csv file

pd.DataFrame.duplicated(df) #check for duplicate data

pd.DataFrame.isnull(df).sum() #check for nulls

df=df[['carat','cut','color','clarity','price']] #reduce the dataframe to only the 4Cs and price

print(df.head()) #print the first 5 rows of the dataframe to confrim the reduction of featuers and target

dict_cut = {"cut": {"Fair":0,"Good":1,"Very Good":2,"Premium":3,"Ideal":4}} #Create a dictonary where the value goes up as the quality goes up
df.replace(dict_cut, inplace=True) #Replace data in cut with their equivalent from the dict_cut

dict_color={"color":{"J":0,"I":1,"H":2,"G":3,"F":4,"E":5,"D":6}} #Create a dictonary where the value goes up as the quality goes up
df.replace(dict_color, inplace=True) #Replace data in color with their equivalent from the dict_color


dict_clar={"clarity":{"I1":0,"SI1":1,"SI2":2,"VS1":3,"VS2":4,"VVS1":5,"VVS2":6,"IF":7}} #Create a dictonary where the value goes up as the quality goes up
df.replace(dict_clar, inplace=True) #Replace data in clarity with their equivalent from the dict_clar


print(df.head()) #print the first 5 rows of the dataframe to confirm that the dictionary replacements have taken place.

df.hist(figsize=[10,10]) #create histograms of the Features and Target data to look at distribution of data.

df.to_csv('//home/derrick/DiamondsClean.csv') #Save the cleaned data

print(df.describe()) #Get the Mean and other important information of each column

mdl_diamond = ols("price ~ carat + cut + color + clarity", data=df).fit() #create the Linear Regression model, using Ordinary Least Squares

print(mdl_diamond.params) #Print the parameters of the model

print(mdl_diamond.summary()) #print model summary
