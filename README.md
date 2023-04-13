# Advanced-Stats-3

This repository contains my solutions to four different problems related to statistical analysis in R. 

# Problem 1: Non-Constant Variance

NIST Alaska Pipeline Data Analysis

Problem Statement

Researchers at National Institutes of Standards and Technology (NIST) collected pipeline data on ultrasonic measurements of the depth of defects in the Alaska pipeline in the field. The depth of the defects were then remeasured in the laboratory. These measurements were performed in six different batches. It turns out that this batch effect is not significant and so can be ignored in the analysis that follows. The laboratory measurements are more accurate than the in-field measurements, but more time consuming and expensive. We want to develop a regression equation for correcting the in-field measurements.

The pipeline data is available in the library faraway

```ruby
install.packages("faraway")
```
Analysis

The analysis involves two parts:

Part 1: Non-constant Variance

We first fit a regression model Lab ~ Field and check for non-constant variance. We observe that the spread of the residuals grows larger and larger as the fitted values increase (cone-shaped). This is a typical sign of non-constant variance. To address this, we apply three different transformations (log, sqrt, and inverse) on both Lab and Field variables and observe the plots of transformed variables. On eyeballing, we see that the log-log transformation is much more linear than the rest. Hence, we construct the model with a log transformation on both sides.

Part 2: Regression Analysis

We fit the regression model on the transformed data using lm function in R and obtain the summary of the model. We also plot the residuals against the fitted values using ggplot. The log-log transformation fixes the problem of non-constant variance, as we observe that the residuals are evenly spread across on both sides of the line and we do not observe any pattern as before.

Conclusion

The regression model with log-log transformation on both sides is the best fit for correcting the in-field measurements. This analysis provides a useful tool for correcting the in-field measurements with greater accuracy and at a lower cost.


# Problem 2: Box Cox Transformation
In this problem, we used the ozone dataset from the faraway library to fit a model for O3 as the response and temp, humidity, and ibh as predictors. We used the Box-Cox plot to find the best transformation on the response.

#Problem 3: Feature Selection

Feature Selection Methods
This task deals with the Boston dataset in the ISLR2 package. We will try to predict the per capita crime rate.

Read this data into an R dataframe and plot them using a scatterplot matrix (pairs) and using ggplot. Describe the dataset. Comment on the correlations between predictors. Based on the pairwise plot or ggplot, can we say any of the predictors are more important?

Fitted a multiple regression model between y and the other variables as predictors. Evaluate the coefficients using summary(), and summarize what you have learned. What are the significance of p-values?

Commented on the interpretation of the coefficients of the predictors. Which predictors are more important? Can we say about the importance of predictors from the coefficient values? (Assume you don't know about feature selection methods for this sub-question.)

Performed feature selection using the following methods by splitting the dataset by random splitting into two parts - 80 % of the dataset as a training dataset and 20 % of the test set. Compare the performance of the 'reduced' models on the test dataset.

Forward Stepwise with p-value threshold of 0.1
Backward Stepwise with p-value threshold of 0.1
Forward Stepwise with AIC
Forward Stepwise with BIC
Forward Stepwise with Mallows Cp



#Problem 4: Cross Validation

In this task, we will perform a regression of the form Y ~ Pd_k=1 Xk and use cross-validation as a method to identify the optimal d. We will implement the cross-validation method.

We first construct our dataset in the following way:

Sample a vector X ∈ R^n where each Xi ∈ U[0, 1]. Each sample point Xi is sampled from the uniform distribution. Construct Y from X using the following equation:

Y = 3 × X^5 + 2 × X^2 + ε

where ε ∈ R^n. Each εi is sampled independently from the N(0,0.5) (normal distribution).

Choose n = 10000.

Split the 10000 points into an 80% training and 20% test split. Use a seed before randomizing to replicate results.

Split the training set into 5 parts and use the five folds to choose the optimal d. The loss function you would implement is the MSE error. You want

# Bias Variance TradeOff

The Bias Variance Tradeoff is a fundamental concept in machine learning. It refers to the tradeoff between the ability of a model to fit the training data well (low bias) and the ability of the model to generalize well to new data (low variance).

In this exercise, we will use 1000 datasets and the function Y ∼ poly(X) to train models of the form Y ∼ poly(X) with different degrees of polynomial (d ∈ [1, 2, . . . , 10]). For each of the models, we will compute the bias and variance while predicting the output at a new test point x = 1.01.

Procedure

Generate 1000 datasets with n = 100 using the function Y ∼ sin(πX) + ε, where ε ∼ N(0, 0.25).
For each dataset, train 10 different models (for d ∈ [1, . . . , 10].) and store the prediction for x = 1.01.
Calculate the bias and variance of the prediction value for each d. Plot the bias and variance as a function of d.

Results

We observe that as we increase the degree of the polynomial, the bias decreases and the variance increases. This is the Bias-Variance tradeoff.

Bias-Variance Tradeoff Plot

As we increase the complexity of the model, the bias decreases as the model can fit the training data better. However, the variance increases, as the model becomes more sensitive to noise and starts to fit the noise in the training data.


Implications

The Bias-Variance tradeoff has important implications in machine learning. Models with low bias tend to be complex and have high variance, while models with low variance tend to be simple and have high bias. Therefore, we need to strike a balance between bias and variance to achieve good generalization performance.

Mitigation

We can mitigate the issues related to the Bias-Variance tradeoff by using regularization techniques such as Ridge Regression or Lasso Regression, which can reduce the variance of the model. We can also use techniques such as cross-validation to tune the hyperparameters of the model and prevent overfitting. Finally, we can use ensemble methods such as bagging and boosting, which combine multiple models to reduce the variance and improve the performance of the model.






