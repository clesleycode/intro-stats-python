Intro to Statistics with Python
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958).

## Table of Contents

- [0.0 Setup](#00-setup)
	+ [0.1 R and R Studio](#01-r-and-r-studio)
	+ [0.2 Packages](#02-packages)
- [1.0 Background](#10-background)
	+ [1.1 Machine Learning](#11-Machine Learning)
	+ [1.2 Data](#12-data)
	+ [1.3 Overfitting vs Underfitting](#13-overfitting-vs-underfitting)
	+ [1.4 Glossary](#14-glossary)
		* [1.4.1 Factors](#141-factors)
		* [1.4.2 Corpus](#142-corpus)
		* [1.4.3 Bias](#143-bias)
		* [1.4.4 Variance](#144-variance)
- [2.0 Data Preparation](#30-data-preparation)
	+ [2.1 dplyr](#31-dplyr)
	+ [2.2 Geopandas](#32-geopandas)
- [3.0 Exploratory Analysis](#30-exploratory-analysis)
- [4.0 Data Visualization](#50-data-visualization)
- [5.0 Machine Learning & Prediction](#50-machine-learning--prediction)
	+ [5.1 Random Forests](#51-random-forests)
	+ [5.2 Natural Language Processing](#52-natural-language-processing)
		* [5.2.1 ANLP](#521-anlp)
	+ [5.3 K Means Clustering](#53-k-means-clustering)
- [6.0 Final Exercise]($60-final-exercise)
- [7.0 Final Words](#60-final-words)
	+ [7.1 Resources](#61-resources)
	+ [7.2 More!](#72-more)

## 0.0 Setup

TThis guide was written in Python 3.5.

### 0.1 Python and Pip

Download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).

### 0.2 Libraries

```

```

## 1.0 Background

The purpose of this tutorial is using your ability to code to help you understand probability and statistics.


### 1.1 Probability

Probability is the study of random events.

### 1.2 Statistics

Statistics is the discipline of using data samples to support claims about populations. Most statistical analysis is based on probability, which is why these pieces are usually presented together.

#### 1.2.1 Data collection 


#### 1.2.2 Descriptive statistics 

Descriptive statistics refers to the generation of statistics that summarize your data concisely and evaluate different ways to visualize data.

#### 1.2.3 Exploratory data analysis

During the process of exploratory data analysis, we look for patterns, differences, and other features that address the questions we are interested in.At the same time we will check for inconsistencies and identify limitations.

#### 1.2.4 Hypothesis testing 

Where we see apparent effects, like a difference between two groups, we will evaluate whether the effect is real, or whether it might have happened by chance.

#### 1.2.5 Estimation 

We will use data from a sample to estimate characteristics of the general population.

### 1.3 Computation

Computation is a tool that is well-suited to quantitative analysis, and computers are commonly used to process statistics. Also, computational experiments are useful for exploring concepts in probability and statistics.


### 1.4 Glossary

#### 1.4.1 Frequency 

Frequency is the number of times a value appears in a dataset

#### 1.4.2 Probability

Probability is the frequency expressed as a fraction of the sample size, n.

## 2.0 Descriptive Statistics 


### 2.1 Mean 

An “average” is one of many summary statistics you might choose to describe the typical value or the central tendency of a sample.

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/mean.png?raw=true "Logo Title Text 1")


### 2.2 Variance

In the same way that the mean is intended to describe the central tendency, variance is intended to describe the <b>spread</b>. 

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/variance.png?raw=true "Logo Title Text 1")

The x<sub>i</sub> - &mu; is called the "deviation from the mean", making the variance the mean multipled by the squared deviation. This is why the square root of the variance, &sigma;, is called the <b>standard deviation</b>.


### 2.3 Distributions

Summary statistics are concise, but dangerous, because they obscure the data. An alternative is to look at the <b>distribution</b> of the data, which describes how often each value appears.


#### 2.3.1 Histograms

The most common representation of a distribution is a histogram, which is a graph that shows the frequency or probability of each value.

Let's say we have the following list: 

``` python
t = [1,2,2,3,1,2,3,2,1,3,3,3,3]
```

To get the frequencies, we can represent this with a dictionary:

``` python
hist = {}
for x in t:
	hist[x] = hist.get(x,0) + 1
```

Now, if we want to convert these frequencies to probabilities, we divide each frequency by n, where n is the size of our original list. This process is called <b>normalization</b>.

``` python
n = float(len(t))
pmf = {}
for x, freq in hist.items():
	pmf[x] = freq / n
```

This normalized histogram is called a PMF, “probability mass function”, which is a function that maps values to probabilities.


#### 2.3.2 Mode

The most common value in a distribution is called the <b>mode</b>.

#### 2.3.3 Shape

The shape just refers to the shape the histogram data forms. Typically, we look for asymetry, or a lack there of.

#### 2.3.5 Outliers

Outliers are values that are far from the central tendency. Outliers might be caused by errors in collecting or processing the data, or they might be correct but unusual measurements. It is always a good idea to check for outliers, and sometimes it is useful and appropriate to discard them.




## 3.0 Cumulative Distribution Functions

### 3.1 Percentiles

The percentile rank is the fraction of people who scored lower than you (or the same). So if you are “in the 90th percentile,” you did as well as or better than 90% of the people who took the exam.

``` python
def percentileRank(scores, your_score):
	count = 0
	for score in scores:
		if score <= your_score:
			count += 1
	percentile_rank = 100.0 * count / len(scores)
	return(percentile_rank)
```

### 3.2 CDFs

The Cumulative Distribution Function (CDF) is the function that maps values to their percentile rank in a distribution.

The following function should look familiar - it's almost the same as percentileRank,
except that the result is in a probability in the range 0–1 rather than a percentile
rank in the range 0–100.

``` python
def cdf(t, x):
	count = 0.0
	for value in t:
		if value <= x:
			count += 1.0
	prob = count / len(t)
	return(prob)
```

## 4.0 Continuous Distributions

The distributions we have used so far are called empirical distributions because they are based on empirical observations, which are necessarily finite samples. The alternative is a continuous distribution, which is characterized by a CDF that is a continuous function (as opposed to a step function).

### 4.1 Exponential Distribution 

Exponential distributions come up when we look at a series of events and measure the times between events, which are called interarrival times. If the events are equally likely to occur at any time, the distribution of interarrival times tends to look like an exponential distribution.

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/exp%20cdf.png?raw=true"Logo Title Text 1")

Here, &lambda; determines the shape of the distribution. 


### 4.2 Pareto Distribution 

The Pareto Distribution is often used to describe phenomena in the natural and social sciences including sizes of cities and towns, sand particles and meteorites, forest fires and earthquakes.


![alt text](https://github.com/lesley2958/stats-programmers/blob/master/pareto%20cdf.png?raw=true "Logo Title Text 1")

















###  Z-Values

Z-value is a measure of standard deviation, i.e. how many standard deviation away from mean is the observed value. For example, the value of z-value = +1.8 can be interpreted as the observed value is +1.8 standard deviations away from the mean. 


### P-Values

Meaningwhile, P values are probabilities. Both these statistics terms are associated with the standard normal distribution. 


### Central Limit Theorem 


### Significance Level


## 5.0 Final Words


### 5.1 Resources

[]() <br>
[]()

### 5.2 Mini Courses

Learn about courses [here](www.byteacademy.co/all-courses/data-science-mini-courses/).

[Python 101: Data Science Prep](https://www.eventbrite.com/e/python-101-data-science-prep-tickets-30980459388) <br>
[Intro to Data Science & Stats with R](https://www.eventbrite.com/e/data-sci-109-intro-to-data-science-statistics-using-r-tickets-30908877284) <br>
[Data Acquisition Using Python & R](https://www.eventbrite.com/e/data-sci-203-data-acquisition-using-python-r-tickets-30980705123) <br>
[Data Visualization with Python](https://www.eventbrite.com/e/data-sci-201-data-visualization-with-python-tickets-30980827489) <br>
[Fundamentals of Machine Learning and Regression Analysis](https://www.eventbrite.com/e/data-sci-209-fundamentals-of-machine-learning-and-regression-analysis-tickets-30980917759) <br>
[Natural Language Processing with Data Science](https://www.eventbrite.com/e/data-sci-210-natural-language-processing-with-data-science-tickets-30981006023) <br>
[Machine Learning with Data Science](https://www.eventbrite.com/e/data-sci-309-machine-learning-with-data-science-tickets-30981154467) <br>
[Databases & Big Data](https://www.eventbrite.com/e/data-sci-303-databases-big-data-tickets-30981182551) <br>
[Deep Learning with Data Science](https://www.eventbrite.com/e/data-sci-403-deep-learning-with-data-science-tickets-30981221668) <br>
[Data Sci 500: Projects](https://www.eventbrite.com/e/data-sci-500-projects-tickets-30981330995)
