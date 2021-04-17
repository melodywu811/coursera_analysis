# Coursera Course Information and Reviews 

This project comprises two main parts. In the first part, I scraped course information from 348 data science courses on [Coursera](https://www.coursera.org/) and applied machine learning models to predict enrollment with different course characteristics (i.e. page views, ratings, difficult levels, outcome highlights). 

In the second part, I scaped 12,000+ reviews from the most popular data science course, [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning), and applied NLP techniques on these textual data to understand the user/learner experience of taking MOOC courses and to further improve the quality of MOOCs and online learning. I also developed a classifier to identify bad reviews from textual data, which course developers can use it trigger review notices for improvement areas.  

The rest of the summary is organized as below:  
1. Background
2. Problem Statement
3. Data Collection and Pre-Processing
4. Exploratory Data Analysis (EDA)
5. Modeling 
6. Intepretation, Conclusions and Recommendations

---
## Background  
### What are MOOCs? 
Massive Open Online Courses (MOOCs) are paid/free online courses available for anyone to enroll (note: some programs and MOOCs might enroll students based on a selective admission, but this is not common). MOOCs provide an affordable and flexible way to learn new skills, advance careers,s and deliver quality educational experiences at scale. 

Source: https://www.mooc.org/ 

### What is Coursera? 
Coursera Inc. is an American MOOC provider founded in 2012 by Stanford University computer science professors Andrew Ng and Daphne Koller. Coursera works with universities and other organizations to offer online courses, certifications, and degrees in a variety of subjects.  

Source: https://about.coursera.org/

---
## Problem Statement
Although MOOCs can provide convenience and flexibility to the learners, it has significant drawbacks. In an article in Science, Dr. Justin Reich and Dr. José A. Ruipérez-Valiente from Massachusetts Institute of Technology (MIT) highlight that "among all MOOC participants, only 3.13 percent completed their courses in 2017-18". 

The most direct way to investigate this issue is to analyze user logs and completion status to identify and prevent dropout. However, there are no Coursera user logs available to the public for predicting dropouts. Therefore, in this project, **the main problem I hope to solve is to understand MOOC learner experience and identify improvement areas by analyzing the course data scraped from Coursera.**

1. By conducting EDA on course information, I uncover characteristics/trends among the data science courses and explore the general landscape of this subject area on Coursera. 
2. By predicting enrollment, I investigate the characteristics that can contribute but the enrollment/popularity of a data science course on Coursera 
3. By conducting NLP EDA, sentiment analysis, and topic modeling on the review data, I examine what are some of the most discussed topics, sentiments of learners in data science on Coursera and identify areas of improvement. 
4. By developing and analyzing the classifiers to differentiate good and bad reviews, I investigate whether the ratings are consistent with review texts and how to use the classifier to generate areas of improvement.    

---
## Data Collection and Pre-Processing
Due to the lack of existing open-source MOOC data, all data used in this project are scraped from the Coursera website. Also, Coursera has terminated giving API access to public users, therefore, the data scraping in this project is done manually using Beautiful Soup and Selenium. There are two parts of data collection: 1) scraping the course information from all data science courses on Coursera and 2) scraping the course review data from one single course (the most popular data science course chose since it has the most number of reviews).   

### Part I: 
When the data were collected, there were 747 courses on Coursera that were labeled in the “Data Science” subject. Some of these courses were cross-listed, for example, the Exploratory Data Analysis course is labeled as both “Data Science” and “Business”. In this report, if a course is cross-listed as Data Science and something else, I still treated it as a Data Science course for simplicity. Some courses have insufficient data, for example, some courses do not report any course outcome data, and I have excluded these courses from the analysis pool. As a result, there were only 348 courses in the EDA and modeling session. 

### Part II:
There are 12,888 reviews (textual data) scraped from the Machine Learning Course (instructed by Andrew Ng from Stanford University), along with its ratings. These textual data are pre-processed using NLP techniques (i.e. tokenization, lemmatization, stopword removal, etc.)

---
## Exploratory Data Analysis (EDA)
The Exploratory Data Analysis (EDA) also comprises two parts and below presents the key findings from EDA. 

### Part I: EDA on the course information data
- Enrollment, recent views, and the number of ratings are highly skewed to the right. 
- All courses are 100% online and have flexible deadlines. 99% of the courses have shareable certificates. 
- Among all courses, 61% of them are courses offered by higher education institutions (as opposed to tech companies or other course providers). This number is even higher among the most popular courses (69%), indicating that students are keener on taking MOOCs offered by higher education institutes. 
- 64% of the courses are part of a specialization. A specialization is a series of courses, for example, the Data Science Specialization by Johns Hopkins University comprises 10 different courses.  
- The three highlighting career outcome metrics are: 1) started a new career, 2) got a tangible career benefit, and 3) got a pay increase. Among all the data science courses analyzed, on average 31% of the learners who completed the courses started a new career, 21%  got a tangible career benefit, and 32% got a pay increase. 
- Enrollment is highly correlated with recent views, the number of ratings, and the number of reviews. The most popular courses a longer than less popular courses. 
- It is surprising that enrollment is positively correlated with course length but is not correlated with career outcomes. 

### Part II: EDA on the course review data (The Machine Learning Course)
- The mean review rating is 4.55, with 5.0 as the max and 3.0 as min. Among all reviews, 96.9% of them rating above 3, which is categorized as “good reviews” in this project. The rest are label as “bad reviews”. 
- The most frequent words are quite different between good and bad reviews. Learners who gave good reviews mention the instructor more and talk more generally about the program/course. On the other hand, learners who gave good reviews talk more about specific things about the program, such as the video quality, math requirements. 
- Sentiment analysis shows that good reviews and bad reviews have very different sentiment distributions. Good reviews generally have higher positive sentiment scores and bad reviews have higher negative sentiment scores. 
- Topic modeling shows that there are some mutual topics discussed among all reviews. One topic is around the pedagogy of the courses, such as the instruction, assignment, lectures. Another topic is technical skills, such as the math, and programming background before taking this course. Another topic is about the general feedback of taking online courses on Coursera. 

---
# Modeling 
This modeling session also comprises two parts. 

### Part I: Predicting enrollment 
The goal of this part is to apply machine learning algorithms to predict enrollment (numeric, continuous) with course information characteristics (e.g., course ratings, recent reviews, course outcomes, etc.). Three models - Decision Tree Classifier, XGBoost, and Linear Regression - are shortlisted from seven models based on their preliminary (untuned) R-square scores and RMSE. Though the linear regression model only provides moderate predictability, it is shortlisted because of its interpretability. Model performance among the shortlisted models after tuning the hyperparameters: 
- R-square score improves to 0.838 with GridSearching the Decision Tree Model.
- R-square score improves to 0.885 with GridSearching the XGBoost Model.
- R-square score improves to 0.883 with PolynomialFeatures and LASSO regularization in the linear regression model. 
- R-square score improves to 0.868 with PolynomialFeatures and Ridge regularization in the linear regression model. 
- By interpreting the linear regression model with PolynomialFeatures and LASSO regularization, the most important features are: `'num_ratings', 'num_reviews', 'num_ratings length', 'num_ratings shareable_certificate', 'num_ratings he_partner', 'num_reviews shareable_certificate', 'length shareable_certificate', 'num_ratings shareable_certificate he_partner', 'num_ratings he_partner^2', 'num_reviews^2 beginner_level', 'num_reviews shareable_certificate^2', 'length^2 he_partner', 'length shareable_certificate^2'`   
All these features and their poly/interaction terms are positively related to enrollment except for `num_reviews^2 beginner_level`. 

### Part II: Course review classifier modeling
The goal of the classifier is to train a model that can classify good and bad reviews based on the texts. As mentioned in the EDA session, class imbalances are a critical issue for this model. Before adjusting class imbalance, both logistic and tree-based models cannot out-perform the baseline (the majority class). These models have a low recall rate of the positive class (only 11% of the bad reviews being identified as bad reviews). 

After adjusting class imbalances by employing different techniques - Undersample More Frequent Class, Oversample Less Frequent Class, SMOTE (Synthetic Minority Over-sampling Technique) - the recall rate of the positive class improves from 0.11 to 0.85. The trade-off is that the recall rate of the negative class (good reviews) reduced from 1.00 to 0.86. The final model employs a logistic regression for its interpretability. This model has an accuracy rate of 0.858, a recall rate of 0.85 for the positive class, and 0.86 for the negative class. 


---
# Intepretation, Conclusions and Recommendations

*work in progress*




