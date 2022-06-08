# Short Title

IBM-AWS Immersion Day Lab 4

# Long Title

Build machine learning models with no code in a collaborative data science environment.

# Author

* [Manoj Jahgirdar](https://developer.ibm.com/profiles/manoj.jahgirdar)
* [Sharath Kumar RK](https://developer.ibm.com/profiles/sharrkum)

# URLs

### Github repo

* https://github.ibm.com/ibm-aws/ibm-aws-immersion-day-lab-4

# Summary

In this Immersion Day Lab, you will build time-series machine learning models and visualize the results using IBM Cloud Pak for Data Jupyter Notebooks, AutoAI and Embedded Dashboard on Amazon Web Services (AWS) Cloud. Developers will learn both **Code** and **No Code** approach to build models and visualize them.  

# Description

This lab demonstrates how to build state of the art predictive models using different techniques. It covers building the models from scratch using code based approach in the Cloud Pak for Data environment which will be helpful for the Data Scientists, ML Engineers & architects. It also covers building accurate predictive models using No-Code approach through guided Machine Learning technique which will be helpful for Data Stewards, analysts & more. Deploying the models for real-time scoring can be done quickly to aid production workloads. 

The intent of this workshop is to educate users about the features of IBM Cloud Pak for Data running on AWS.

Once you complete the code pattern, you will learn to:

* Build a state of the art Long Short Term Memory(LSTM) prediction models using IBM Cloud Pak for Data Jupyter Notebook
* Visualize the actual vs predicted values in IBM Cloud Pak for Data Cognos Dashboard Embedded
* Build and compare different time-series models with no code in IBM Cloud Pak for Data AutoAI Experiments

# Flow

![architecture](doc/source/images/architecture.png)

1. Pre-processed datasets from Lab 1 are loaded into an Amazon S3 bucket
2. The datasets from the S3 bucket are read in Jupyter Notebooks
3. Different models are built and evaluated in Jupyter Notebooks and the final prediction data is stored back into S3 bucket
4. The datasets from the S3 bucket is copied into Watson Studio Project and loaded into AutoAI. Different models are built and compared in AutoAI with no code
5. The prediction data produced by the Jupyter Notebook models stored in S3 bucket is read by Cognos Dashboard Embedded to visualize the data in the form of interactive dashboard  

# Instructions

> Find the detailed steps in the [README](https://github.ibm.com/ibm-aws/ibm-aws-immersion-day-lab-4/blob/main/README.md) file.

1. Setup a S3 Bucket
2. Setup a project in Cloud Pak for Data
    * 2.1. Create a Project
    * 2.2. Create a Connection to S3
3. Code Approach: Build Prediction Models with Watson Studio
    * 3.1. About the Notebooks
    * 3.2. Run LSTM Notebook 1term-memory-lstm-model)
    * 3.3. Run LSTM Notebook 2term-memory-lstm-model)
    * 3.4. Run Decision Tree Notebook
4. No Code: Build Prediction Models with IBM Cloud Pak for Data AutoAI
5. Visualize the Predictions in IBM Cloud Pak for Data Cognos Embedded Dashboard
    * 5.1. Setup Cognos Embedded Dashboard
    * 5.2. Analyze Cognos Embedded Dashboard

# Components and services

* IBM Cloud Pak for Data Watson Studio
* IBM Cloud Pak for Data Jupyter Notebooks
* IBM Cloud Pak for Data AutoAI
* IBM Cloud Pak for Data Cognos Embedded Dashboard
