import pandas as pd
import seaborn as sns

medical = pd.read_csv('C:/users/dscha/Downloads/medical_raw_data.csv', index_col=0) #create the medical dataframe
medical.info() #print out summarized data, including Column name, count of non-null entries, and data type

medical.duplicated() #find duplicated records

medical.drop_duplicates() #drop any duplicated records
medical.isnull().sum() #find the total number of null records for each column
#create numeric columns for each yes/no column, then replace yes with 1 and no with 0
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
# End of numeric encoding

medical.hist(figsize =[20,20]) # Create histograms of each column for distribution models

medical['Age'].fillna(medical['Age'].mean(), inplace =True) #Replace null values in Age with the mean age.
medical['Children'].fillna(medical['Children'].median(), inplace=True) #Replace null values in the number of Children with the median number.
medical['Income'].fillna(medical['Income'].median(), inplace=True) #Replace null values in income with the median value for income.
medical['Overweight'].fillna(medical['Overweight'].median(), inplace=True) #Replace null values in Overweight with the median value
medical['Anxiety'].fillna(medical['Anxiety'].median(), inplace=True) #Replace null values in Anxiety with the median value
medical['Initial_days'].fillna(medical['Initial_days'].median(), inplace=True) #Replace the null values for the initial length of stay with the median value.
medical['Soft_drink']=medical['Soft_drink'].fillna(medical['Soft_drink'].mode()[0]) # Replace null values in Number of soft drinks had daily with the mode value.

pd.set_option('display.max_rows', None) #set the max rows displayed to none, so that all column values can be seen.
medical.isnull().sum() #Show a count of null values per column, after cleaning this should come back with all 0s

medical.hist(figsize=[20,20]) #Rerun the histograms to make sure that distribution of data didn't change.
medical.to_csv(r'C:\Users\dscha\Downloads\D206\medical_data_cleaned.csv') #Export cleaned data to a csv.
