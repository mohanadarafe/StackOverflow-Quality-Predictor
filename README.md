# Stack Overflow Quality Predictor

## Setup
In order to run the code, make sure you have Conda installed in your machine in order to use our environment. From there, run the following commands

```
conda env create --name stack --file=environment.yml
conda activate stack
```

## Run Spark
Please make sure you are running the script from the **correct directory**!
```
cd pyspark
python run_pipeline.py
```

## Run Dask
Please make sure you are running the script from the **correct directory**!
```
cd dask
python run.py
```

# Abstract
StackOverflow is every developer's best friend. There are millions of users actively asking their questions, unfortunately, some questions are better than others. How can StackOverflow find a way to filter good questions from bad questions & perhaps suggest users to re-formulate a question before submission. Using big data analytics & machine learning, this project will find a solution to distinguish high quality & low quality questions. We will be comparing big data technologies & study their performance with the same data set. We will compare both Spark & Dask and analyse them based on various metrics such as time of execution, CPU usage & more.

# I. Introduction
StackOverflow is a question and answer site that connects a community of programmers in order to help each other. Having over 10 million registered users & over 16 million questions ( mid 2018 ), analyzing the quality of questions and giving feedback to the author is a must. This can contribute to an improvement in the comprehensibility of the questions thus targeting more users and improving overall the quality of StackOverflow's user experience. As a result, if StackOverflow is able to improve and control the information that is being posted on the platform and perform quality checks this can only benefit them and benefit the users. The objectives for the project will be separated in two parts. In the first part, we will preprocess our data using both Spark & Dask. In the second part, we will train our dataset using a Naive Bayes classifier. The objective of this is to study the effects a big data technology will have on our models. Is Spark faster than Dask? These are some of the things we will look at. Both Spark & Dask will take the raw dataset & transform them into the same desired format. Once we have that data ready to go, we will train a Naive Bayes classifier that can predict the quality of a question asked based on its title & content. Currently, Stack Overflow uses heuristics to determine if a question is not recommended to be posted. There is no official documentation of the heuristics they use, but it’s definitely a feature that is currently on their website. 

The project idea itself is not ”unique”. There is research done in the field where artificial intelligence practitioners build a Stack Overflow question quality detection algorithm. For example, this paper written by various teachers at the University of Szeged in Hungary share their research in attempting to detect the quality of questions in Stack Overflow using NLP techniques [1]. In our project, we will be facing the same challenge but with the goal to compare big data technologies.

II. Material & Methods
The dataset consists of 60,000 questions from 2016 to 2020. They are separated in both a training & validation set which have 45,000 & 15,000 questions respectively. We will use both datasets & create a test set out of them in order to get performance metrics. As for the columns, it’s fairly straight forward, there are 6 features in total. Namely, the question ID, title, body, tags, creation date & quality. As for the quality, the dataset is partitioned in three categories: high quality, low quality edited & low quality closed. High quality questions have a total score of 30+ & no edits. Low quality edits have negative scores, community edits but they remain open. Low quality closed questions closed by the community with no edits. Cleaning & processing the data will require natural language processing techniques such as tokenization, removing stop-words & more. You can find the Kaggle dataset [here](https://www.kaggle.com/imoore/60k-stack-overflow-questions-with-quality-rate?select=train.csv)

<img width="954" alt="data" src="https://user-images.githubusercontent.com/34899555/115128434-b92ead00-9fab-11eb-86ba-f2c07539ee5b.png">

Using both Spark & Dask's machine learning libraries, we are going to preprocess the data with the APIs provided to us by the respective libraries as well as the classifiers available. The tasks involved can be divided in four stages for each tool:

- Read and parallelize questions
- Clean & preprocess data
- Build a Bag-of-Word model
- Train data

From there, we can compare using graphs various metrics from each tool. Namely, we will firstly look at how the models performed. Then, we will extract information from each run by looking at time of execution, CPU usage, memory usage, process scheduling & level of parallelism.

# IV. Results & Discussion
In order to compare Dask and Spark we decided to analyze predefined metrics that will help us better understand what is happening under the hood and why one technology performs better than the other. The metrics that we were interested in are mainly regrouped into two different categories. The first category contains the system information which in our case is our local machine (16GB RAM, 4 Cores) such as CPU, memory and disk in use. We collected this data before and after each function call and recorded the execution time of each function. The second category contains the information of the processes that were running during the execution time such as the name of the process, creation time, up-time, time on CPU, memory usage as well as the number of threads that the process is using. When doing our experiment, we granted Dask and Spark access to all the cores of our machine in order to maximize performance.

The graphs above compare the execution time of Spark and Dask. It displays the time it spends on each operation. Spark 2.4.5 was faster than Dask **(201s vs 230s)** considering the fact that Spark is doing a larger number of operations than Dask as shown in the x-axis which represents the operations performed. We can also observe that training the models and building the bag of word models are the most expensive operations with 120s for Spark and 200s for Dask. The reason why Dask is performing poorly comes from the fact that Dask ML's API requires costly operations. For example, the [CountVectorizer API](https://ml.dask.org/modules/generated/dask_ml.feature_extraction.text.CountVectorizer.html) which is used to convert questions of a BoW model requires the input to be a Dask Bag. Converting from a Dask DataFrame to a Dask Array is a costly operation & it shows in the graph.

Interestingly, we ran the software with a higher version of Spark, namely, version 3.1.1 & it came to our surprise that is ran significantly faster. As you can see from the graph below, the execution time of Spark 3.1.1 is a staggering **352.42 seconds**. We believe this phenomenon is due to the fact that there is a higher level of parallelism in Spark 2.4.5.

Regarding the CPU & memory usage, Dask is performing better by using less resources totalling a maximum of 14& of the CPU where Spark is using a maximum of 23&. Moreover, Dask uses a **maximum of 5.6%** of the memory where Spark uses a **minimum of 5.7%**. Spark 3.1.1 & Spark 2.4.5 make similar use of memory. Spark 3.1.1 uses significantly more of the CPU by reaching 35% of consumption.

In addition, we collected metrics on the processes. Spark 2.4.5 uses three Python processes and one Java process versus Dask which uses just the one Python process. We can see that the Python processes of the Spark execution are triggered at different moments in time during the execution of the tasks. Regarding Spark 2.4.5, the process _python3.6-0_ was alive during the whole execution of the program & it used 6 threads. It had a CPU time of 1.7s where it used 0.4& of the CPU and 0.18% of the memory. The other Python processes were also alive during the whole execution time but only used 1 thread each and their consumption of the CPU was insignificant (less than 1%). The Java process used 99% of the CPU & 8.13% of the memory but utilized a significant amount of threads, 220 in total. Spark 3.1.1 performed similarly to its peer Spark 2.4.5 but its Java process consumed **549%** of the CPU. Dask had a Python process that used a massive 99.70% of the CPU, 3.9% of the memory with 12 threads.

Spark and Dask both have pros & cons. Firstly, Spark was developer friendly, the extensive documentation and community helped us develop this experiment in a smooth way. Secondly, Dask was harder to develop since most of its API's are not very flexible in terms of the format of the data required. In fact, the hardest part was converting data to adapt to Dask ML's requirements. In terms of performance, Spark performed better considering the higher number of tasks that it had to accomplish. Meanwhile, Dask was slower considering the little amount of tasks it performed compared to Spark. As we mentioned, Dask used less resources than Spark which begs the following question: which do you prioritize, speed or resource efficiency? For us, we would use Spark in another experiment & not Dask due to its considerable limitations. In a future experiment, we would try to run the softwre in one common high performing computer & observe any changes.

# Figures
![figures_1](https://user-images.githubusercontent.com/34899555/115128730-e2e8d380-9fad-11eb-84ba-e0e01d5aa453.png)

![figures_2](https://user-images.githubusercontent.com/34899555/115128804-702c2800-9fae-11eb-9eaf-cd221c5d303c.png)

# References
[1] Tóth, László & Nagy, Balázs & Janthó, Dávid & Vidacs, Laszlo & Gyimóthy, Tibor. (2019). Towards an Accurate Prediction of the Question Quality on Stack Overflow Using a Deep-Learning-Based NLP Approach. 10.5220/0007971306310639.

### Authors
[Mohanad Arafe](https://github.com/mohanadarafe)

[Robert Beaudenon](https://github.com/RobertBeaudenon)
