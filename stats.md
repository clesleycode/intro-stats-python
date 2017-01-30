Intro to Statistics with Python
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958).

## Table of Contents

- [0.0 Setup](#00-setup)
	+ [0.1 Python and Pip](#01-python-and-r-pip)
	+ [0.2 Libraries](#02-libraries)
- [1.0 Background](#10-background)
	+ [1.1 Probability](#11-probability)
	+ [1.2 Statistics](#12-statistics)
		* [1.2.1 Data Collection](#121-data-collection)
		* [1.2.2 Descriptive Statistics](#122-descriptive-statistics)
		* [1.2.3 Exploratory Data Analysis](#123-exploratory-data-analysis)
		* [1.2.4 Hypothesis Testing](#124-hypothesis-testing)
		* [1.2.5 Estimation](#125-estimation)
	+ [1.3 Computation](#13-computation)
	+ [1.4 Glossary](#14-glossary)
		* [1.4.1 Frequency](#141-frequency)
		* [1.4.2 Probability](#142-probability)
		* [1.4.3 Oversampling](#143-oversampling)
		* [1.4.4 Summary Statistic](#144-summary-statistic)
		* [1.4.5 Statistically Significant](#145-statistically-signifcant)
		* [1.4.6 Central Tendency](#146-central-tendency)
		* [1.4.7 Frequentist Statistics](#147-frequenist-statistics)
		* [1.4.8 Bayesian Statistics](#148-bayesian-statistics)
- [2.0 Descriptive Statistics](#20-descriptive-statistics)
	+ [2.1 Mean](#21-mean)
	+ [2.2 Variance](#22-variance)
	+ [2.3 Distributions](#23-distributions)
		* [2.3.1 Histograms](#231-histograms)
		* [2.3.2 Mode](#232-model)
		* [2.3.3 Shape](#233-shape)
		* [2.3.4 Outliers](#234-outliers)
- [3.0 Cumulative Distribution Functions](#30-cumulative-distribution-functions)
	+ [3.1 Percentiles](#31-percentiles)
	+ [3.2 CDFs](#32-cdfs)
	+ [3.3 Interquartile Range](#33-interquartile-range)
- [4.0 Sampling Distributions](#40-sampling-distributions)
	+ [4.1 Normal Distribution](#41-normal-distribution)
	+ [4.2 Exponential Distribution](#42-exponential-distributions)
	+ [4.3 Pareto Distribution](#43-pareto-distribution)
	+ [4.4 Poisson Distribution](#44-poisson-distribution)
		* [4.4.1 Code](#441-code)
- [5.0 Probability](#50-probability)
	+ [5.1 Probability Rules](#51-probability-rules)
	+ [5.2 Binomial Distribution](#52-binomial-distribution)
	+ [5.3 Bayes's Theorem](#55-bayes-theorem)
		* [5.3.1 What is a Sampling Distribution?](#551-what-is-a-sampling-distribution)
- [6.0 Operations on Distributions](#60-operations-on-distributions)
	+ [6.1 Skewness](#61-skewness)
		* [6.1.1 Pearson’s Median Skewness Coefficient](#611-pearsons-median-skewness-coefficient)
- [7.0 Hypothesis Testing](#70-hypothesis-testing)
	+ [7.1 Testing a difference in means](#71-testing-a-difference-in-means)
	+ [7.2 Choosing a threshold](#72-choosing-a-threshold)
- [8.0 Estimation](#80-estimation)
	+ [8.1 Outliers](#81-outliers)
	+ [8.2 Mean Squared Error](#82-mean-squared-error)
- [9.0 Correlation](#90-correlation)
	+ [9.1 Covariance](#covariance)
	+ [9.2 Correlation](#92-correlation)
- [10.0 Mini Courses](#100-mini-courses)

## 0.0 Setup

TThis guide was written in Python 3.5.

### 0.1 Python and Pip

Download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).

### 0.2 Libraries

```
pip3 install scipy
pip3 install numpy
```

## 1.0 Background

The purpose of this tutorial is to use your ability to code to help you understand probability and statistics.

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

#### 1.4.3 Oversampling

Oversampling is the technique of increasing the representation of a subpopulation in order to avoid errors due to small sample sizes.

#### 1.4.4 Summary Statistics

A summary statistic is the result of a computation that reduces a dataset to a single number (or at least a smaller set of numbers) that captures some characteristic of the data.

#### 1.4.5 Statistically Significant

An apparent effect is statistically significant if it is unlikely to occur by chance.

#### 1.4.6 Central Tendency

The central tendency is a characteristic of a sample or population, or the most average value. 

#### 1.4.7 Frequentist Statistics

Frequentist Statistics tests whether an event occurs or not. It calculates the probability of an event in the long run of the experiment (i.e the experiment is repeated under the same conditions to obtain the outcome).

#### 1.4.8 Bayesian Statistics

Bayesian statistics is a mathematical procedure that applies probabilities to statistical problems. It provides people the tools to update their beliefs in the evidence of new data.


## 2.0 Descriptive Statistics

### 2.1 Mean 

An “average” is one of many summary statistics you might choose to describe the typical value or the central tendency of a sample.

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/mean.png?raw=true "Logo Title Text 1")

In Python, the mean would look like this: 

``` python
def Mean(t):
    return(float(sum(t)) / len(t))
```

Alternatively, you can use built-in functions from the numpy module: 

``` python
import numpy
np.mean([1,4,3,2,6,4,4,3,2,6])
```

### 2.2 Variance

In the same way that the mean is intended to describe the central tendency, variance is intended to describe the <b>spread</b>. 

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/variance.png?raw=true "Logo Title Text 1")

The x<sub>i</sub> - &mu; is called the "deviation from the mean", making the variance the mean multipled by the squared deviation. This is why the square root of the variance, &sigma;, is called the <b>standard deviation</b>.

Using the mean function we created above, we'll write up a function that calculates the variance: 

``` python
def Var(t, mu=None):
    if mu is None:
        mu = Mean(t)
    # compute the squared deviations and returns their mean.
    dev2 = [(x - mu)**2 for x in t]
    var = Mean(dev2)
    return(var)
```
Once again, you can use built in functions from numpy instead:

```
numpy.var([1,3,3,6,3,2,7,5,9,1])
```

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

#### 2.3.4 Outliers

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

The following function should look familiar - it's almost the same as percentileRank, except that the result is in a probability in the range 0–1 rather than a percentile rank in the range 0–100.

``` python
def cdf(t, x):
	count = 0.0
	for value in t:
		if value <= x:
			count += 1.0
	prob = count / len(t)
	return(prob)
```

Alternatively, you can use numpy to find the percentile. 

``` python
import numpy
numpy.percentile([1,42,53,23,12,3,35,2], 50)
```

This code returns the 50th percentile, e.g median.

### 3.3 Interquartile Range

Once you have computed a CDF, it's easy to compute other summary statistics.The median is just the 50th percentile. The 25th and 75th percentiles are often used to check whether a distribution is symmetric, and their difference, which is called the interquartile range, measures the spread.



## 4.0 Sampling Distributions

The distributions we have used so far are called empirical distributions because they are based on empirical observations, which are necessarily finite samples. The alternative is a continuous distribution, which is characterized by a CDF that is a continuous function (as opposed to a step function).

### 4.1 Normal Distribution

The normal distribution, also called Gaussian, is the most commonly used because it describes so many phenomena (at least approximately). The normal distribution has many properties that make it amenable for analysis, but the CDF is not one of them.

Unlike the other distributions we will look at, there is no closed-form expression for the normal CDF. Instead, we write it in terms of the error function, erf(x). 

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/normal%20cdf.png?raw=true "Logo Title Text 1")

Now, using the scipy module, we can create the CDF for a Normal Distribution:

``` python
from scipy.special import erf

def StandardNormalCdf(x):
    return (erf(x / root2) + 1) / 2)

def NormalCdf(x, mu=0, sigma=1):
    return(StandardNormalCdf(float(x - mu) / sigma))
```
Notice we imported `er` from `scipy.special`. This CDF, when plotted, looks like:

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/normal%20distr%20cdf%20plot.png?raw=true "Logo Title Text 1")

### 4.2 Exponential Distribution 

Exponential distributions come up when we look at a series of events and measure the times between events, which are called interarrival times. If the events are equally likely to occur at any time, the distribution of interarrival times tends to look like an exponential distribution.

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/exp%20cdf.png?raw=true"Logo Title Text 1")

Here, &lambda; determines the shape of the distribution. The mean of an exponential distribution is 1/&lambda;, whereas the median is usually ln(2)/&lambda;. This results in a distribution that looks like:

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/exp%20cdf%20plot.png?raw=true "Logo Title Text 1")


### 4.3 Pareto Distribution 

The Pareto Distribution is often used to describe phenomena in the natural and social sciences including sizes of cities and towns, sand particles and meteorites, forest fires and earthquakes.

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/pareto%20cdf.png?raw=true "Logo Title Text 1")

Here, x<sub>m</sub> and &alpha; determine the location and shape of the distribution. Specifically x<sub>m</sub> is the minimum possible value. This ends up looking something like this: 

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/pareto%20cdf%20plot.png?raw=true "Logo Title Text 1")


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

### 5.1 Probability Rules

Generally speaking, P(A and B) = P(A) P(B), but this is not always true. 

If two events are mutually exclusive, that means that only one of them can happen, so the conditional probabilities are 0: P(A|B) = P(B|A) = 0. In this case it is easy to compute the probability of either event:
P(A or B) = P(A) + P(B)

### 5.2 Binomial Distribution 

If I roll 100 dice, the chance of getting all sixes is (1/6)<sup>100</sup>. And the chance of getting no sixes is (5/6)<sup>100</sup>. Those cases are easy, but more generally, we might like to know the chance of getting k sixes, for all values of k from 0 to 100. The answer is the binomial distribution, which has this PMF:

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/binomial%20pmf.png?raw=true "Logo Title Text 1")

where n is the number of trials, p is the probability of success, and k is the
number of successes. The binomial coefficient is pronounced “n choose k”, and it can be computed
recursively like this:

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/binomial%20coeff.png?raw=true "Logo Title Text 1")

In Python, this looks like: 

``` python
def Binom(n, k, d={}):
    if k == 0:
        return(1)
    if n == 0:
        return(0)
    try:
        return(d[n, k])
    except KeyError:
        res = Binom(n-1, k) + Binom(n-1, k-1)
        d[n, k] = res
        return(res)
```


### 5.3 Bayes's Theorem

Bayes’s theorem is a relationship between the conditional probabilities of two events. A conditional probability, often written P(A|B) is the probability that Event A will occur given that we know that Event B has occurred. It's represented as follows:

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/bayes.png?raw=true "Logo Title Text 1")

Bayes theorem is what allows us to go from a sampling distribution and a prior distribution to a posterior distribution. 


#### 5.3.1 What is a Sampling Distribution?

A sampling distribution is the probability of seeing a given data point, given our parameters (&theta;). This is written as p(X|&theta;). For example, we might have data on 1,000 coin flips, where 1 indicates a head.

In python, this might look like: 

``` python
import numpy as np
data_coin_flips = np.random.randint(2, size=1000)
np.mean(data_coin_flips)
```

As we said in the previous section, a sampling distribution allows us to specify how we think these data were generated. For our coin flips, we can think of our data as being generated from a Bernoulli Distribution. 

Therefore, we can create samples from this distribution like this:

``` python
bernoulli_flips = np.random.binomial(n=1, p=.5, size=1000)
np.mean(bernoulli_flips)
```

Now that we have defined how we believe our data were generated, we can calculate the probability of seeing our data given our parameters. Since we have selected a Bernoulli distribution, we only have one parameter, p. 

We can use the PMF of the Bernoulli distribution to get our desired probability for a single coin flip. Recall that the PMF takes a single observed data point and then given the parameters (p in our case) returns the probablility of seeing that data point given those parameters. 

For a Bernoulli distribution it is simple: if the data point is a 1, the PMF returns p. If the data point is a 0, it returns (1-p). We could write a quick function to do this:

``` python
def bern_pmf(x, p):
	if x == 1:
		return(p)
	elif x == 0:
		return(1 – p)
	else:
		return("Value Not in Support of Distribution")
```

We can now use this function to get the probability of a data point give our parameters. You probably see that with p = .5 this function always returns .5

``` python
print(bern_pmf(1, .5))
print(bern_pmf(0, .5)) 
```
More simply, we can also use the built-in methods from scipy:

``` python
import scipy.stats as st
print(st.bernoulli.pmf(1, .5))
print(st.bernoulli.pmf(0, .5))
```



## 6.0 Operations on Distributions

### 6.1 Skewness

Skewness is a statistic that measures the asymmetry of a distribution. Given a sequence of values, x<sub>i</sub>, the sample skewness is

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/skewness.png?raw=true "Logo Title Text 1")

You might recognize m<sub>2</sub> as the mean squared deviation (or variance);m<sub>3</sub> is the mean cubed deviation.

Negative skewness indicates that a distribution “skews left". It extends farther to the left than the right. Positive skewness indicates that a distribution skews right.

Because outliers can have a disproportionate effect on g<sub>1</sub>, another way to evaluate the asymmetry of a distribution is to look at the relationship between the mean and median. Extreme values have more effect on the mean than the median, so in a distribution that skews left, the mean is less than the median.

#### 6.1.1 Pearson’s Median Skewness Coefficient

Pearson’s median skewness coefficient is an alternative measure of skewness that explicitly captures the relationship between the mean, &mu;, and the median, &mu;<sub>1/2</sub>. It's particularly useful because it's robust, meaning it is <b>not</b> sensitive to outliers.

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


### 8.1 Outliers

Using the sample mean to estimate &mu; is fairly intuitive, but suppose we introduce outliers. One option is to identify and discard outliers, then compute the sample
mean of the rest. Another option is to use the median as an estimator.

### 8.2 Mean Squared Error

If there are no outliers, the sample mean minimizes the mean squared error (MSE). If we play the game many times, and each time compute the error &#772; - &mu;, the sample mean minimizes: 

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/mse.png?raw=true "Logo Title Text 1")


## 9.0 Correlation

Now, we'll look at relationships between variables. <b>Correlation</b> is a description of some kind of relationship.

### 9.1 Covariance

Covariance is a measure of the tendency of two variables to vary together. If we have two series, X and Y, their deviations from the mean are

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/covariance.png?raw=true "Logo Title Text 1")

where &mu;<sub>X</sub> is the mean of X and &mu;<sub>Y</sub> is the mean of Y. If X and Y vary together, their deviations tend to have the same sign. If we multiply them together, the product is positive when the deviations have the same sign and negative when they have the opposite sign. So adding up the products gives a measure of the tendency to vary together.

Therefore, covariance is the mean of these two products:

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/cov%20final.png?raw=true "Logo Title Text 1")

Note that n is the length of the two series, so they have to be the same length.

``` python
def Cov(xs, ys, mux=None, muy=None):
    """Computes Cov(X, Y).

    Args:
        xs: sequence of values
        ys: sequence of values
        mux: optional float mean of xs
        muy: optional float mean of ys

    Returns:
        Cov(X, Y)
    """
    if mux is None:
        mux = thinkstats.Mean(xs)
    if muy is None:
        muy = thinkstats.Mean(ys)

    total = 0.0
    for x, y in zip(xs, ys):
        total += (x-mux) * (y-muy)

    return(total / len(xs))
```

### 9.2 Correlation

One solution to this problem is to divide the deviations by &sigma;, which yields standard scores, and compute the product of standard scores.

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/pearson%20coeff.png?raw=true "Logo Title Text 1")

Pearson’s correlation is always between -1 and +1. The magnitude indicates the strength of the correlation. If p = 1 the variables are perfectly correlated. The same is true if p = -1. It means that the variables are negatively correlated.

It's important to note that Pearson's correlation only measures <b>linear</b> relationships. 

Using the mean, varainces, and covariance methods above, we can write a function that calculates the correlation. 

``` python 
import math
def Corr(xs, ys):
    xbar = Mean(xs)
    varx = Var(xs)
    ybar = Mean(ys)
    vary = Var(ys)

    corr = Cov(xs, ys, xbar, ybar) / math.sqrt(varx * vary)
    return(corr)
```

## 10.0 Mini Courses

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
