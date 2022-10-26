# VAERS: COVID-19 Vaccine Adverse Reaction Analysis

View the full Python code [here](https://github.com/dallas-hutch/VAERS/blob/main/VAERS_Analysis.ipynb).

## Table of Contents
[Introduction](#introduction)

[Results at a Glance](#results-at-a-glance)

[Part 2. Data Cleaning](#2-data-cleaning)

[Part 3. Exploratory Data Analysis](#3-exploratory-data-analysis)

[Part 4. Regression Modeling](#4-regression-modeling)

[Findings](#findings)

## Introduction
The dataset comes from the VAERS (Vaccine Adverse Event Reporting System) created by the CDC and FDA. It contains patient level data on adverse reactions from the COVID-19 vaccine. Headed into the project, I wanted to see if I could classify whether or not an individual may or may not experience severe side-effects based on factors such as age, gender, vaccine manufacturer, and medical history using machine learning algorithms. Secondly, I wanted to see if there was a relationship between number of days it took to show symptoms from vaccination and the number of total symptoms someone was experiencing. This was accomplished using linear regression and correlation tests. During the analysis, I noticed a trend where older individuals seemed to be more likely to have severe side-effects which led to the third research question. In that, is there a significant difference in age between those patients who experienced severe side-effects and those who did not?

## Results at a Glance
<b>Research Question 1. Can I classify which patients will be more likely to experience severe side effects of the Covid-19 vaccine?</b>
- Based on the factors of age, gender, vaccine manufacturer, and medical history, I was able to classify whether or not a patient would experience severe side effects with an accuracy of 74% and AUC (area under the curve) of 0.80. This tells us the model is able to effectively distinguish between the two groups, those who will experience severe side effects and those who won't. The three models compared were Gradient Boosting, Random Forest, and Logistic Regression.

<img src="https://github.com/dallas-hutch/Covid19-Vaccine-Reaction-Analysis/blob/main/images/gradientboosting_classificationreport.JPG" width=50% height=50%> <img src="https://github.com/dallas-hutch/Covid19-Vaccine-Reaction-Analysis/blob/main/images/auc_curvecomparison.JPG" width=40% height=40%>

<b>Research Question 2. Is there a relationship between the number of days for symptoms to show among patients and the number of symptoms caused by the vaccine? In that if there is a lag from when the vaccine is administered to when symptoms show, does that affect symptom count?</b>
- There is a extremely slight, inverse relationship between number of days it takes to show symptoms from vaccination and the total number of symptoms a patient experiences. That is, as the gap between vaccination and onset date increases, symptom count decreases. Both variables were tested for normality and found to be non-Gaussian. The Spearman's and Kendall's Tau rank tests both supported rejecting the null hypothesis in this case.

<img src="https://github.com/dallas-hutch/Covid19-Vaccine-Reaction-Analysis/blob/main/images/regression_significancetesting.JPG" width=50% height=50%> <img src="https://github.com/dallas-hutch/Covid19-Vaccine-Reaction-Analysis/blob/main/images/lin_regression_fit.JPG" width=40% height=40%>

<b>Research Question 3. Is there a significant difference in age between patients who experienced severe side effects from the vaccine and patients who did not (those with no symptoms or non-severe)? </b>
- There is a statistically significant difference in age between patients who experienced severe side effects and patients who did not. Again, normality was tested for both groups age distributions and found to be non-Gaussian. Then, a Mann-Whitney U test identified a significant difference between the two distributions, which supports rejecting the null hypothesis. In the distribution plot below, we can see patients who experienced severe side effects are more likely to be older than their counterparts.

<img src="https://github.com/dallas-hutch/Covid19-Vaccine-Reaction-Analysis/blob/main/images/mannwhitneyu_test.JPG" width=50% height=50%> <img src="https://github.com/dallas-hutch/Covid19-Vaccine-Reaction-Analysis/blob/main/images/age_dist_plot.JPG" width=40% height=40%>
