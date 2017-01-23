Intro to Statistics with Python
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958).

## Table of Contents

- [0.0 Setup](#00-setup)
	+ [0.1 R and R Studio](#01-r-and-r-studio)
	+ [0.2 Packages](#02-packages)
- [1.0 Background](#10-background)
	+ [1.1 Probability](#11-probability)
	+ [1.2 Statistics](#12-statistics)
		* [1.2.1 Data Collection](#121-data-collection)
		* [1.2.2 Descriptive Statistics](#122-descriptive-statistics)
		* [1.2.3 Exploratory Data Analysis](#123-exploratory-data-analysis)
		* [1.2.4 Hypothesis Testing](#124-hypothesis-testing)
		* [1.2.5 Estimation](#125-estimation)
	+ [1.3 Computation](#13-computation)
- [10.0 Final Words](#100-final-words)
	+ [10.1 Resources](#101-resources)
	+ [10.2 Mini Courses](#72-mini-courses)

## 0.0 Setup

TThis guide was written in Python 3.5.

### 0.1 Python and Pip

Download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).

### 0.2 Libraries

```
pip3 install
```

## 1.0 Background

The purpose of this tutorial is using your ability to code to help you understand probability and statistics.


### 1.1 Probability

Probability is the study of random events - the study of how likely it is that some event will happen.

### 1.2 Statistics

Statistics is the discipline of using data samples to support claims about populations. Most statistical analysis is based on probability, which is why these pieces are usually presented together.

#### 1.2.1 Data Collection 

Data collection is the process of gathering and measuring information on targeted variables in an established systematic fashion, which allows you to answer relevant questions and evaluate outcomes.

Many times, our data will come from simulation. In the classic example question, "Is a coin toss fair?", we'll do exactly that.


#### 1.2.2 Descriptive Statistics 

Descriptive statistics refers to the generation of statistics that summarize your data concisely and evaluate different ways to visualize data.

#### 1.2.3 Exploratory Data Analysis

During the process of exploratory data analysis, we look for patterns, differences, and other features that address the questions we are interested in. At the same time we will check for inconsistencies and identify limitations.

#### 1.2.4 Hypothesis Testing 

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

Here, x<sub>m</sub> and &infty; determine the location and shape of the distribution. Specifically x<sub>m</sub> is the minimum possible value.


### 4.3 Normal Distribution

The normal distribution, also called Gaussian, is the most commonly used because it describes so many phenomena, at least approximately.

### 4.4 Poisson Distribution

The Poisson distribution can be applied to systems with a large number of possible events, each of which is rare.

A discrete random variable X  is said to have a Poisson distribution with parameter &lambda; > 0, if, for k = 0, 1, 2,..., the probability mass function of X  is given by:

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/poisson%20pmf%20.png?raw=true "Logo Title Text 1")

The positive real number &lambda; is equal to the expected value of X and its variance, so &lambda; = E(X) = Var(X).

To calculate poisson distribution we need two variables:

1. Poisson random variable (x): Poisson Random Variable is equal to the overall REMAINING LIMIT that needs to be reached

2. Average rate of success(rate_of_success): 
 

#### 4.4.1 Code

Scipy is a python library that is used for Analytics, Scientific Computing, and Technical Computing. Using the `stats.poisson` module we can easily compute poisson distribution of a specific problem.

Using scipy, we can calculate the poisson distribution as follows: 

proabability_reached = float(1-scipy.stats.distributions.poisson.cdf(poisson random variable-1, rate_of_success)) * 100

## 5.0 Probability

Probability is a real value between 0 and 1 that is intended to be a quantitative measure corresponding to the qualitative notion that some things are more likely than others.

The “things” we assign probabilities to are <b>called events</b>. If E represents an event, then P(E) represents the probability that E will occur. A situation where E might or might not happen is called a trial.


### 5.5 Bayes's Theorem

Bayes’s theorem is a relationship between the conditional probabilities of two events. A conditional probability, often written P(A|B) is the probability that Event A will occur given that we know that Event B has occurred. It's represented as follows:

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/bayes.png?raw=true "Logo Title Text 1")


## 6.0 Operations on Distributions

### 6.1 Skewness

Skewness is a statistic that measures the asymmetry of a distribution. Given a sequence of values, x<sub>i</sub>, the sample skewness is

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/skewness.png?raw=true "Logo Title Text 1")



## 7.0 Hypothesis Testing

A statistical hypothesis is a hypothesis that is testable on the basis of observing a process that is modeled via a set of random variables. The underlying logic is similar to a proof by contradiction. To prove a mathematical statement, A, you assume temporarily that A is false. If that assumption leads to a contradiction, you conclude that A must actually be true.

Similarly, to test a hypothesis like, “This effect is real,” we assume, temporarily,
that is is not. That’s the <b>null hypothesis</b>. Based on that assumption,
we compute the probability of the apparent effect. That’s the <b>p-value</b>. If the
p-value is low enough, we conclude that the null hypothesis is unlikely to
be true.


### 7.1 Testing a difference in means

One of the easiest hypotheses to test is an apparent difference in mean between
two groups.

### 7.2 Choosing a threshold

In hypothesis testing we have to worry about two kinds of errors.

1. False Positives are when we accept a hypothesis that is actually false; that is, we consider an effect significant when it was actually due to chance.

2. False Negatives are when we reject a hypothesis that is actually true; that is, we attribute an effect to chance when it was actually real.


## 8.0 Estimation 

Up until now we have used the symbol &mu; for both the sample mean and the mean parameter, but now we will distinguish them, using x&#772;‌ for the sample mean. Previously, we've just assumed that x&#772;‌ = &mu;, but now we will go through the actual process of estimating &mu;. This process is called estimation, and the statistic we used (the sample mean) is called an estimator.

Using the sample mean to estimate &mu; is fairly intuitive, but suppose we introduce outliers.

## 9.0 Correlation


###  Z-Values

Z-value is a measure of standard deviation, i.e. how many standard deviation away from mean is the observed value. For example, the value of z-value = +1.8 can be interpreted as the observed value is +1.8 standard deviations away from the mean. 


### P-Values

Meaningwhile, P values are probabilities. Both these statistics terms are associated with the standard normal distribution. 


### Central Limit Theorem 


### Significance Level


## 10.0 Final Words


### 10.1 Resources

[]() <br>
[]()

### 10.2 Mini Courses

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
