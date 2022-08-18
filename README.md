# VAERS: COVID-19 Vaccine Adverse Reaction Analysis
This is an analysis I did for my Informatics class. The dataset comes from the VAERS (Vaccine Adverse Event Reporting System) created by the CDC and FDA. It contains patient level data on adverse reactions from the COVID-19 vaccine. Headed into the project, I wanted to see if I could classify whether or not an individual may or may not experience severe side-effects based on factors such as age, gender, vaccine manufacturer, and medical history using machine learning algorithms. Secondly, I wanted to see if there was a relationship between number of days it took to show symptoms from vaccination and the number of total symptoms someone was experiencing. This was accomplished using linear regression and correlation tests. During the analysis, I noticed a trend where older individuals seemed to be more likely to have severe side-effects which led to the third research question. In that, is there a significant difference in age between those patients who experienced severe side-effects and those who did not? Check out the Jupyter notebook for the results!
- Based on the factors of age, gender, vaccine manufacturer, and medical history, I was able to classify whether or not a patient would experience severe side effects with Gradient Boosting performing the best of the three models.
- There is a extremely slight, inverse relationship between number of days it takes to show symptoms from vaccination and the total number of symptoms a patient experiences. That is, as the gap between vaccination and onset date increases, symptom count decreases.
- There is a statistically significant difference in age between patients who experienced severe side effects and patients who did not.

<iframe src="https://www.slideshare.net/slideshow/embed_code/key/neN3vYOg0FMof7?hostedIn=slideshare&page=upload" width="476" height="400" frameborder="0" marginwidth="0" marginheight="0" scrolling="no"></iframe>
