# Capstone

Project Name: Multiple Linear Regression and Deep Neural Networks on Market Data

Research question: Can a multiple linear regression model and DNN be constructed
based solely on the research data?

My Hypothesis: A regression model and DNN Model can be constructed from the market
 data.

Data collected from 
https://www.kaggle.com/datasets/amirhosseinmirzaie/diamonds-price-dataset

Results:
The Linear Regression model showed that carat was the primary driver behind diamond
price, followed by clarity, color and finally cut.  The weights of the 4Cs,
along with the intercept is below:

Intercept: -5551.64
carat: 8515.03
cut: 182.12
color: 285.13

The Deep Neural Network was able to accurately predict the price of diamonds as
shown in the Scatterplot Output.png, and had a Mean Absolute Error of 307.00