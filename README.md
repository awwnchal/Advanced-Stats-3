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



