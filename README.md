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
Research Question 1. Can I classify which patients will be more likely to experience severe side effects of the Covid-19 vaccine?
- Based on the factors of age, gender, vaccine manufacturer, and medical history, I was able to classify whether or not a patient would experience severe side effects with Gradient Boosting performing the best of the three models.

<img src="https://github.com/dallas-hutch/Covid19-Vaccine-Reaction-Analysis/blob/main/images/gradientboosting_classificationreport.JPG" width=50% height=50%>

<img src="https://github.com/dallas-hutch/Covid19-Vaccine-Reaction-Analysis/blob/main/images/auc_curvecomparison.JPG" width=40% height=40%>

- There is a extremely slight, inverse relationship between number of days it takes to show symptoms from vaccination and the total number of symptoms a patient experiences. That is, as the gap between vaccination and onset date increases, symptom count decreases.

<img src="https://github.com/dallas-hutch/Covid19-Vaccine-Reaction-Analysis/blob/main/images/regression_significancetesting.JPG" width=50% height=50%>

<img src="https://github.com/dallas-hutch/Covid19-Vaccine-Reaction-Analysis/blob/main/images/lin_regression_fit.JPG" width=40% height=40%>

- There is a statistically significant difference in age between patients who experienced severe side effects and patients who did not.

<img src="https://github.com/dallas-hutch/Covid19-Vaccine-Reaction-Analysis/blob/main/images/mannwhitneyu_test.JPG" width=50% height=50%>

<img src="https://github.com/dallas-hutch/Covid19-Vaccine-Reaction-Analysis/blob/main/images/age_dist_plot.JPG" width=40% height=40%>
