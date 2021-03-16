# Stack Overflow Quality Predictor

## Abstract
StackOverflow is every developer's best friend. There are millions of users actively asking their questions, unfortunately, some questions are better than others. How can StackOverflow find a way to filter good questions from bad questions & perhaps suggest users to re-formulate a question before submission. Using big data analytics & machine learning, this project will find a solution to distinguish high quality & low quality questions. Also, we will be comparing big data technologies & study their performance with the same dataset. We will explore if certain big data technologies are more performing than others based on various metrics such as time & output produced. 

## Introduction
StackOverflow is a question and answer site that connects a community of programmers in order to help each other. Having over 10 million registered users, and over 16 million questions ( mid 2018 ), analyzing the quality of questions and giving feedback to the author is a must. This can contribute to an improvement in the understandability of the questions  thus targeting more users and improving overall the quality of the stack overflows user experience. As a result, if Stackoverflow is able to improve and control the information that is being posted on the platform and perform quality checks this can only benefit them and benefit the users. The objectives for the project will be separated in two parts. In the first part, we will preprocess our data using both Spark & Dask. In the second part, we will train our dataset using various machine learning models. The objective of this is to study the effects a big data technology will have on our models. Is Spark faster than Dask? These are some of the things we will look at. Both Spark & Dask will take the raw dataset & transform them into the same desired format. Once we have that data ready to go, we will build a machine learning model that can predict the quality of a question asked based on its title & content. Currently, Stack Overflow uses heuristics to determine if a question is not recommended to be posted. There is no official documentation of the heuristics they use, but it’s definitely a feature that is currently on their website. The project idea itself is not ”unique”. There is research done in the field where artificial intelligence practitioners build a Stack Overflow question quality detection algorithm. For example, this paper written by various teachers at the University of Szeged in Hungary share their research in attempting to detect the quality of questions in Stack Overflow using NLP techniques [1]. In our project, we will be facing the same challenge but with the goal to compare big data technologies. 

## Materials & Methods
The dataset consists of 60,000 questions from 2016 to 2020. They are separated in both a training & validation set which have 45,000 & 15,000 questions respectively. We will use both datasets & create a test set out of them in order to get performance metrics. As for the columns, it’s fairly straight forward, there are 6 features in total. Namely, the question ID, title, body, tags, creating date & quality. As for the quality, the dataset is partitioned in three categories: high quality, low quality edited & low quality closed. High quality questions have a total score of 30+ & no edits. Low quality edits have negative scores, community edits but they remain open. Low quality closed questions closed by the community with no edits. It’s worth mentioning that the body is posted as HTML text. There is no doubt overall that natural language processing will be required when analyzing the dataset. You can find the Kaggle dataset [here](https://www.kaggle.com/imoore/60k-stack-overflow-questions-with-quality-rate?select=train.csv)

We are going to experiment with a Naive Bayes Classifier using heuristics & Natural Language Processing.

Tasks:
- Extract the data and build the model (build a probabilistic model from the training set)
- Build a vocabulary by parsing the csv of all the words it contains in the questions
- For each word, compute their frequencies (BoW model)
- For each word, compute their probabilities of each Quality Type class ( high quality, low quality edited & low quality closed.) (P(Wi|ClassA))

Once we have fully extracted the data, we will pass our data to our classifiers that will train & test the Stack Overflow dataset. For now, we will implement a Naive Bayes Classifier from scratch. Once we test its performance, we may add more classifiers from scikit-learn based on the performance.

Experiments with the classifier:
- We can create a new model by filtering out stop words.
- We can create a new model by filtering out some words from the vocabulary based on the frequency of the term. (e.g the words that have low frequency).
- Much more... we can create multiple heuristics & experiment them!

Since we will implement all of this twice, once in Dask & once in Spark (if we have time in plain python also). We can compare using graphs the runtime performance for each technology. For the models used, we will implement a Naive Bayes classifier. Also a note since we will be implementing the Naive Bayes Classifier we will only leverage sklearn to evaluate the performance such as accuracy, recall & precision.


## Citations
[1] Tóth, László & Nagy, Balázs & Janthó, Dávid & Vidacs, Laszlo & Gyimóthy, Tibor. (2019). Towards an Accurate Prediction of the Question Quality on Stack Overflow Using a Deep-Learning-Based NLP Approach. 10.5220/0007971306310639. 

### Authors
[Mohanad Arafe](https://github.com/mohanadarafe)

[Robert Beaudenon](https://github.com/RobertBeaudenon)
