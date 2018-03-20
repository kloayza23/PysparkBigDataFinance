# Analsis Big data for Finance data set with pyspark #

In this work going to analyze how use pyspark in an big data anlysis.

### How to Begin this Work ###

In the files of this work exist a file called "FinalProject.ipynb", in this file only you have to run this box of code,
each box has a header explain what is the meaning.

This python jupyter file has been tested many times, offering quality in the code

### Data Set ###

The data set used in this work has 887379 observations and 75 features to evaluate model.

Download Dataset: https://www.kaggle.com/wendykan/lending-club-loan-data

### EXPERIMENTS ###

Data source was loaded through pyspark function, where all variables are string type, where it has passed by some filters 
to get a worthy data. It is a pipeline composed by Encoding -> Features creator and logistic. 
In the process encode and convert to float was used the user define function, instead to remove features, was used a process 
of correlations that let remove variable more similar, keeping few features, where it was split in training and test set for 
be used in the model.

### Contribution ###

The main contribution for this work is the use of the Big data tools for a credit score modeling, using a predictive modeling, in this research have been used the logistic regression model for predict the future example.

### Who do I talk to? ###

* This work is usefull to people where wish to learn pyspark, the first steps
* This work too is usefull to people in the financial environment, where wish to learn how to handle big data set.

### Set Up ###

* Import Libraries
* Load Data Set
* Clean data
* Check for duplications	
* Encoding: Convert the categorical variable in numeric variable
* Application Encoding Process in Data Set
* Export to CSV to find out Size in File
* Convert Variable to Float
* Discretization: Put values in range.
* Decode Variables
* Descriptive Statistics
* Test Balance of Class Loan Status
* Correlations between features
* Visualization
* Splitting into training and testing
* Predicting Loan Status through Logistic Regression
* Evaluating the performance of the model
* Saving the model
* Save the Pipeline
* Save the model


