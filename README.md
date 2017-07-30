# Hotel Recommendation for Online Travel Agencies
**Cem KILICLI - JULY, 2017**
### Capstone Project
This project is a capstone project for Mef University - Mcs. Big Data Analytics


## Introduction
Expedia is an online travel agency that offers thousands of properties (hotels, hostels, etc...) all over the world for travelers. With this massive amount of inventory, it is crucial to understand and offer properties which will suit to personal preference of the users. Expedia wants to take user experience to the next level by providing personalized hotel recommendations to their users. Currently Expedia uses search parameters to adjust their recommendations, but there is enough customer specific data to personalize these recommendations for each user. In this project, we have taken up the challenge to contextualize customer data and predict the likelihood a user will stay at 100 different hotel groups.

The train/test datasets used for this project have been provided by Expedia, via Kaggle, and they contain 23 features capturing the logs of customer behaviour. There is an additional dataset containing 149 features, which covers the destination related information. The goal is to build a machine learning model to predicts best recommendations based on hotel clusters using the user search and event attributes that is provided by Expedia. As a part of this project I have tried to come up with a clustering model via using below given algorithms;

1.	Gaussian Naive Bayes
2.	Decision Tree
3.	K-Nearest Neighbors
4.	K Means
5.	Linear Discriminant Analysis
6.	Multi-Layer Perceptron
7.	Multinomial Logistic Regression
8.	Random Forest
9.	Ensemble Learning Methods (with above given algorithms)
 -	Voting Classifier (hard)
 -	Bagging

Firstly, I begin with exploring the data finding anomalies and correlations within the data. Since the data that is provided is big enough I have enriched the data and pre-process and reshape in a way that is processable by machine learning algorithms.
