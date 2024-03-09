# Google-Stock-Analyzer


Intro
I programmed a Google Stock Analyzer, which uses machine learning techniques to make economic predictions about Google’s stock. I used a dataset from kaggle, a machine learning community, showing Google’s closing prices for the past 15 years on the S&P 500 stock index. I basically dropped the closing price column of the dataset and used machine learning to predict Google’s stock closing prices for the past 15 years.


Libraries
First I imported all the necessary machine learning libraries. These libraries include pandas, numpy, matplotlib.pyplot, train_test_split, random forest regressor, mean squared error, seaborn, linear regressor, decision tree regressor, KNeighbors regressor, elastic net regressor, and finally XGBoost.

Pandas: 
Pandas allows us to analyze big data and make conclusions based on statistical theories. Pandas can clean messy data sets, and make them readable and relevant. Relevant data is very important in data science.

Numpy: 
Numpy is a general-purpose array-processing package

Matplotlib.pyplot: 
Matplotlib is the graphics library for data visualization in Python.

Seaborn: 
Seaborn aims to make visualization the central part of exploring and understanding data. It provides dataset-oriented APIs so that I can switch between different visual representations for the same variables for a better understanding of the dataset.


Stuff from sklearn is below—---------------------------------------------------------------------

Train test split:
Train test split is a model validation procedure that allows you to simulate how a model would perform on new/unseen data

Mean squared error:
mean squared error (MSE) is one of many metrics you could use to measure your model’s performance.

Linear regressor: 
Use linear regression to predict the output based on a linear combination of input features. it is basically a line of best fit like in algebra y=mx=b

Decision tree regressor: 
Decision tree regression observes features of an object and trains a model in the structure of a tree to predict data in the future to produce meaningful continuous output.

Random forest regressor:
Use Random Forest Regression in order to build multiple decision trees and average their predictions. I am using it to prevent overfitting.

KNeighbors regressor: 
Use K Nearest Neighbors Regression to predict the output based on the average of the k-nearest neighbors in the feature space. Basically, it predicts to the point that it is closest to.

Elastic net regressor: 
Use Elastic Net regression (combines L1 (lasso, and L2 (ridge) regularization) in order to both prevent overfitting, and effectively performing feature selection.
L1 (Lasso): 
Adds a regularization term to the linear regression objective function, preventing overfitting.
L2 (Ridge): 
Similar to ridge regression but uses the absolute values of the coefficients. It can lead to sparse models (some coefficients become exactly zero), effectively performing feature selection.

XGBoost: 
Use Gradient Boosting Regression (XGBoost) to build a series of weak learners (typically decision trees) sequentially, with each one correcting the errors of the previous one. Basically, you are correcting the decision trees from the decision tree regression and the random forest regression


Load the dataset
I found a dataset on Kaggle, a machine learning community, that contains all of Google’s stock closing prices for the past 15 years. So I loaded it in, check for null values, applied the datetime function to make it the correct data, and plotted the closing prices overtime.


Correlation heatmap
I then needed to find the relations between each column of the dataset. I used the correlation heat map from matplotlib.pyplot and seaborn to find the correlations. If x goes up, y goes up. Correlation heat maps essentially show the 1 to 1 relationships for the different columns of the dataset.

-0.45 shows not a very strong correlation, so I do not focus on that relationship at all

a high negative and a high positive correlation value shows the relationships and is helpful when trying to understand data.
therefore, the volume seems not have much of a relationship with any other values in the data


Drop closing prices and break off the dataset into train and test model
To do my predictions, I first had to drop the closing prices column as that is the column I need to predict.
After dropping it, I broke off the dataset into train and test models with x_train, y_train and x_test, y_test.

After this, I then printed the number of rows and columns to confirm that I dropped the closing prices column




Regression
I used regression models such as Linear, Decision Tree, and Random Forest to make predictions. To enhance predictive accuracy, I also used Elastic Net Regression, which balances L1 and L2 regularization. The integration of XGBoost adds an extra layer of sophistication to my predictive modeling. Visualizations, correlation heatmaps, and feature selection will help stakeholders to understand the market dynamics.

I started off using Linear Regression. I split the dataset into the training dataset and the testing dataset. The goal of this was to train and test the model to get the RMSE to see the fitting. RMSE stands for root mean squared error. The lower the RMSE is, the more fitted the model is. However, if the RMSE is zero, then the model has overfitted the data. Overfitting means you overtrained the data. Overfitting is bad for predictions because of the increased error on test data, sensitivity to noise (data that is not needed), the increased risk of false discoveries, and it is resource intensive (takes up a lot of processing power and will make you computer lag).	After finding the RMSE fo the training and testing sets for the Linear regression model, I found out that the fitting for Linear regression was pretty good.

I then used another regression model, called Decision tree regression. The whole process of using these regression models is splitting the data into the training and testing sets, fitting them, predicting, and finding the RMSE. After doing this for the decision tree regression, I found out that the decision tree regression wasn’t good for this project because it overfitted the data. It overfitted the data because it is zero.

I then used Random forest regression. The RMSE of the training and testing set were actually fitted the model quite well. It got as close to zero as possible without being zero. I liked this a lot, but I decided to try out other regression models as well to see if there was a better model.

I then used KNN, K nearest neighbors regression. The RMSE for this one wasn’t very I then used Elastic net regression. This model’s RMSE was quite close, but it was still higher than the Random forest regression model. 

Lastly, I used XGboost regression. To use this, I had to convert the data to DMatrix format for the XGboost regressor to work. I then specified the XGboost parameters, trained it, and made it give predictions for the training and testing dataset. It was the second closest to zero. So ultimately, I decided to use the Random forest regression model for the predictions

The reason the RMSE is higher for the testing set is that I are putting more data into the training set than the testing set. The split is 80% to 20%. I are doing this so that the testing set can actually test the model without giving it the answer.

Coming back to random forest regression, I made params, a dictionary of hyperparameters, to reduce overfitting. I then created the random forest regressor with specified parameters,  trained it, and made predictions for the past 15 years.

I then needed to plot the predictions. So I then created the random forest regressor with specified parameters, trained it, and made the predictions. I then converted the index of “df” to datetime.
After all of that, I was finally ready to plot the predictions. First I used the matplotlib.pyplot library to plot the original closing prices. Then I plotted my predicted closing prices.
The result is a graph showing the original closing prices compared to my predicted closing prices over 15 years. It turns out that I was quite accurate.

One weakness that this code has is that it was hard to reduce the overfitting. To combat this, I used hyperparameters. The purpose of hyperparameters is to control the learning process and reduce overfitting.
