Intro to Statistics with Python
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958).

## Table of Contents

- [0.0 Setup](#00-setup)
	+ [0.1 Python and Pip](#01-python-and-pip)
	+ [0.2 Libraries](#02-libraries)
- [1.0 Background](#10-background)
	+ [1.1 Probability](#11-probability)
	+ [1.2 Statistics](#12-statistics)
		* [1.2.1 Data Collection](#121-data-collection)
		* [1.2.2 Descriptive Statistics](#122-descriptive-statistics)
		* [1.2.3 Exploratory Data Analysis](#123-exploratory-data-analysis)
		* [1.2.4 Hypothesis Testing](#124-hypothesis-testing)
		* [1.2.5 Estimation](#125-estimation)
	+ [1.3 Glossary](#13-glossary)
		* [1.3.1 Frequency](#131-frequency)
		* [1.3.2 Probability](#132-probability)
		* [1.3.3 Oversampling](#133-oversampling)
		* [1.3.4 Statistically Significant](#145-statistically-significant)
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
 	+ [4.5 Operations on Distributions](#45-operations-on-distributions)
		* [4.5.1 Skewness](#451-skewness)
		* [4.5.2 Pearson’s Median Skewness Coefficient](#452-pearsons-median-skewness-coefficient)
- [5.0 Probability](#50-probability)
	+ [5.1 Probability Rules](#51-probability-rules)
	+ [5.2 Binomial Distribution](#52-binomial-distribution)
	+ [5.3 Bayes's Theorem](#55-bayes-theorem)
		* [5.3.1 What is a Sampling Distribution?](#551-what-is-a-sampling-distribution)
- [6.0 Estimation](#60-estimation)
	+ [6.1 Outliers](#61-outliers)
	+ [6.2 Mean Squared Error](#62-mean-squared-error)
- [7.0 Hypothesis Testing](#70-hypothesis-testing)
	+ [7.1 Testing a difference in means](#71-testing-a-difference-in-means)
	+ [7.2 Choosing a threshold](#72-choosing-a-threshold)
	+ [7.3 Significance Level](#73-significance-level)
	+ [7.4 Steps](#74-steps)
	+ [7.5 Example](#75-example)
		* [7.5.1 Hypothesis](#651-hypothesis)
		* [7.5.2 Significance Level](#752-significance-level)
		* [7.5.3 Computation](#753-computation)
		* [7.5.4 Hypotheses](#754-hypotheses)
- [8.0 Correlation](#80-correlation)
	+ [8.1 Covariance](#81-covariance)
	+ [8.2 Correlation](#82-correlation)
- [9.0 Mini Courses](#90-mini-courses)

## 0.0 Setup

TThis guide was written in Python 3.5.

### 0.1 Python and Pip

Download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).

### 0.2 Libraries

Let's install the modules we'll need for this tutorial. Open up your terminal and enter the following commands to install the needed python modules:

```
pip3 install scipy
pip3 install numpy
pip3 install math
```

## 1.0 Background

The purpose of this tutorial is to use your ability to code to help you understand probability and statistics.

### 1.1 Probability

Probability is the study of random events - the study of how likely it is that some event will happen.

### 1.2 Statistics

Statistics is the discipline that uses data to support claims about populations. Most statistical analysis is based on probability, which is why these pieces are usually presented together.

#### 1.2.1 Data Collection 

Data collection is the process of gathering data to answer relevant questions and evaluate outcomes.

#### 1.2.2 Descriptive Statistics 

Descriptive statistics refers to statistics that summarize your data concisely and evaluate different ways to visualize data.

#### 1.2.3 Exploratory Data Analysis

During the process of exploratory data analysis, we look for patterns, differences, and other features that address the questions we are interested in. At the same time we will check for inconsistencies and identify limitations.

#### 1.2.4 Hypothesis Testing 

When we evaluate a possible cause-and-effect relationship, like a difference between two groups, we will evaluate whether the effect is real or whether it might have happened by chance.

#### 1.2.5 Estimation 

Estimation is what allows us to use data from a sample to estimate characteristics of the general population.

### 1.3 Glossary

Here is some common terminology that we'll encounter throughout the workshop:

#### 1.3.1 Frequency 

Frequency is the number of times a value appears in a dataset

#### 1.3.2 Probability

Probability is the frequency expressed as a fraction of the sample size, n.

#### 1.3.3 Oversampling

Oversampling is the technique of increasing the representation of a subpopulation to avoid errors due to small sample sizes.

#### 1.3.4 Statistically Significant

An apparent effect is statistically significant if it is unlikely to occur by chance.


## 2.0 Descriptive Statistics

Descriptive Statistics are the basic operations used to gain insights on a set of data.

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
import numpy as np
np.mean([1,4,3,2,6,4,4,3,2,6])
```

### 2.2 Variance

In the same way that the mean is intended to describe the central tendency, variance is intended to describe the <b>spread</b>. 

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/variance.png?raw=true "Logo Title Text 1")

The x<sub>i</sub> - &mu; is called the "deviation from the mean", making the variance the squared deviation multiplied by 1 over the number of samples. This is why the square root of the variance, &sigma;, is called the <b>standard deviation</b>.

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

### 3.1 Percentile Rank & Percentiles

The percentile rank is the fraction of people who scored lower than you (or the same). So if you are “in the 90th percentile,” you did as well as or better than 90% of the people who took the exam.

``` python
def percentileRank(scores, your_score):
	count = 0
	for score in scores:
		if score <= your_score:
			count += 1
	percentile_rank = 100.0 * count / len(scores)
	return(percentile_rank)

percentileRank([1,42,53,23,12,3,35,2], 17.5)
```

Alternatively, we can use the `scipy` module to retrieve the percentile rank! 
``` python
from scipy import stats
stats.percentileofscore([1,42,53,23,12,3,35,2], 17.5)
```

Both of these output the 50th percentile since 17.5 is the median!

Now, what if we want the reverse? So instead of what percentile a value is, we want to know what value is at a given percentile. In other words, now we want the inputs and outputs to be switched. Luckily, this is available with `numpy`:

``` python
import numpy as np
np.percentile([1,42,53,23,12,3,35,2], 50)
```

This code returns the 50th percentile, e.g median, `17.5`.

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

### 3.3 Interquartile Range

Once you have computed a CDF, it's easy to compute other summary statistics. The median is just the 50th percentile. The 25th and 75th percentiles are often used to check whether a distribution is symmetric, and their difference, which is called the interquartile range, measures the spread.


## 4.0 Sampling Distributions

The distributions we have used so far are called empirical distributions because they are based on empirical observations, which are necessarily finite samples. The alternative is a continuous distribution, which is characterized by a CDF that is a continuous function (as opposed to a step function).

### 4.1 Normal Distribution

The normal distribution, also called Gaussian, is the most commonly used because it describes so many scenarios. Despite its wide range of applicability, CDFs with the normal distribution are non-trivial compared to other distributions.

Unlike the other distributions we will look at, there is no closed-form expression for the normal CDF. Instead, we write it in terms of the error function, erf(x). 

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/normal%20cdf.png?raw=true "Logo Title Text 1")

Now, using the scipy module, we can create the CDF for a Normal Distribution:

``` python
from scipy.special import erf

def NormalCdf(x):
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

``` python
proabability_reached = float(1-scipy.stats.distributions.poisson.cdf(poisson random variable-1, rate_of_success)) * 100
```

### 4.5 Operations on Distributions

#### 4.5.1 Skewness

Skewness is a statistic that measures the asymmetry of a distribution. Given a sequence of values, x<sub>i</sub>, the sample skewness is

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/skewness.png?raw=true "Logo Title Text 1")

You might recognize m<sub>2</sub> as the mean squared deviation (or variance). m<sub>3</sub> is the mean cubed deviation.

Negative skewness indicates that a distribution “skews left". It extends farther to the left than the right. Positive skewness indicates that a distribution skews right.

To find this value, you can use `scipy`:

``` python
import scipy
scipy.stats.skew([1,3,3,6,3,2,7,5,9,1])
```

Which gets us a measure of `0.592927061281571`, meaning it's skewed to the right.

Because outliers can have a disproportionate effect on g<sub>1</sub>, another way to evaluate the asymmetry of a distribution is to look at the relationship between the mean and median. Extreme values have more effect on the mean than the median, so in a distribution that skews left, the mean is less than the median.

Take the example from above:

```
[1,3,3,6,3,2,7,5,9,1]
```

The median of this list is `3`, whereas the mean is `4`. With these two values, you can confirm that it skews to the right. 

#### 4.5.2 Pearson’s Median Skewness Coefficient

Pearson’s median skewness coefficient is an alternative measure of skewness that explicitly captures the relationship between the mean, &mu;, and the median, &mu;<sub>1/2</sub>. It's particularly useful because it's robust, meaning it is <b>not</b> sensitive to outliers.

The equation is as follows: 
```
P = (3 * (X - Med))/s
```
where X is the mean, Med is the median, and s is the standard deviation. 

For `[1,3,3,6,3,2,7,5,9,1]`, the mean is 21, the median is 17.5, and the standard deviation is `18.808`. If we plug these values in, we can a pearson median coefficient of `0.5582781958205234`, meaning it's right skewed.


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

where n is the number of trials, p is the probability of success, and k is the number of successes. The binomial coefficient is pronounced “n choose k”, and it can be computed
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

## 6.0 Estimation 

Up until now we have used the symbol &mu; for both the sample mean and the mean parameter, but now we will distinguish them, using x&#772; for the sample mean. Previously, we've just assumed that x&#772; = &mu;, but now we will go through the actual process of estimating &mu;. This process is called estimation, and the statistic we used (the sample mean) is called an estimator.


### 6.1 Outliers

Using the sample mean to estimate &mu; is fairly intuitive, but suppose we introduce outliers. One option is to identify and discard outliers, then compute the sample mean of the rest. Another option is to use the median as an estimator.

### 6.2 Mean Squared Error

If there are no outliers, the sample mean minimizes the mean squared error (MSE). If we iterate through a dataset, and each time compute the error x&#772; - &mu;, the sample mean minimizes: 

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/mse.png?raw=true "Logo Title Text 1")



## 7.0 Hypothesis Testing

A statistical hypothesis is a hypothesis that is testable on the basis of observing a process that is modeled via a set of random variables. The underlying logic is similar to a proof by contradiction. To prove a mathematical statement, A, you assume temporarily that A is false. If that assumption leads to a contradiction, you conclude that A must actually be true.

Similarly, to test a hypothesis like, “This effect is real,” we assume, temporarily, that is is not. That’s the <b>null hypothesis</b>, which is what you typically want to disprove. Based on that assumption, we compute the probability of the apparent effect. That’s the <b>p-value</b>. If the p-value is low enough, we conclude that the null hypothesis is unlikely to
be true.

### 7.1 Z-Values, P-Values & Tables

These are associated with standard normal distributions. Z-values are a measure of how many standard deviations away from mean is the observed value. P-values are the probabilities, which you can retrieve from its associated z-value in a [z-table](http://www.stat.ufl.edu/~athienit/Tables/Ztable.pdf). 

We've already reviewed how to retrieve the p-value, but how do we get the z-value? With the following formula:

![alt text](https://github.com/ByteAcademyCo/stats-programmers/blob/master/z%20value.png?raw=true "Logo Title Text 1")

where x is your data point, &mu; is the mean and &sigma; is the standard deviation. 


### 7.2 Central Limit Theorem

The central limit theorem allows us to understand the behavior of estimates across repeated sampling and conclude if a result from a given sample can be declared to be “statistically significant".

The central limit theorem tells us exactly what the shape of the distribution of means will be when we draw repeated samples from a given population.  Specifically, as the sample sizes get larger, the distribution of means calculated from repeated sampling will approach normality. 

Let's take a look at an example: Here, we have data of 1000 students of 10th standard with their total marks. Let's take a look at the frequency distribution of marks: 

![alt text](https://github.com/ByteAcademyCo/stats-programmers/blob/master/clt-hist.png?raw=true "Logo Title Text 1")

This is clearly an unnatural distribution. So what can we do? 

Let's take a sample of 40 students from this data. That makes for 25 total samples we can take (1000/40 = 25). The actual mean is 48.4, but it's very unlikely that every sample of 40 will have this same mean. 

If we take a large number of samples and compute the means and then make a probability histogram on these means, we'll get something similar to:

![alt text](https://github.com/ByteAcademyCo/stats-programmers/blob/master/clt-samp.png?raw=true "Logo Title Text 1")

You can see that distribution resembles a normally distributed histogram. 

### 7.3 Significance Level

Significance Tests allow us to see whether there is a significant relationship between variables. It gives us an idea of whether something is likely or unlikely to happen by chance. 

### 7.4 Steps

The initial step to hypothesis testing is to actually set up the Hypothesis, both the NULL and Alternate.  

Next, you set the criteria for decision. To set the criteria for a decision, we state the level of significance for a test. Based on the level of significance, we make a decision to accept the Null or Alternate hypothesis.

The third step is to compute the random chance of probability. Higher probability has higher likelihood and enough evidence to accept the Null hypothesis.

Lastly, you make a decision. Here, we compare p value with predefined significance level and if it is less than significance level, we reject Null hypothesis, else we accept it.


### 7.5 Example

Blood glucose levels for obese patients have a mean of 100 with a standard deviation of 15. A researcher thinks that a diet high in raw cornstarch will have a positive effect on blood glucose levels. A sample of 36 patients who have tried the raw cornstarch diet have a mean glucose level of 108. Test the hypothesis that the raw cornstarch had an effect or not.

#### 7.5.1 Hypothesis

First, we have to state the hypotheses. We set our NULL Hypothesis to be the glucose variable = 100 since that's the known fact. The alternative is that the glucose variable is greater than 100. 


#### 7.5.2 Significance Level

Unless specified, we typically set the significance level to 5%, or `0.05`. Now, if we figure out the corresponding z-value from the [z-table](http://www.stat.ufl.edu/~athienit/Tables/Ztable.pdf), we'll see that it corresponds to `1.645`. This is now the z-score cut off for significance level, meaning the area to the right (or z-scores higher than 1.645) is the rejection hypothesis space. 

#### 7.5.3 Computation

Now, we can compute the random chance probability using z scores and the z-table. Recall the formula from earlier, z = (x - &mu;)/ &sigma;. Now, before we go into computing, let's overview the difference between standard deviation of the mean and standard deviation of the distribution. 

When we want to gain a sense the precision of the mean, we calculate what is called the <i>sample distribution of the mean</i>. Assuming statistical independence, the standard deviation of the mean is related to the standard deviation of the distribution with the formula &sigma;<sub>mean</mean> = &sigma / &radic;N. 

With that knowledge in mind, we've been given the standard deviation of the distribution, but we need the standard deviation of the mean instead. So before we begin calculating the z value, we plug in the values for the formula above. Then we get &sigma;<sub>mean</sub> = 15 / &radic;36, or `2.5`.

Now we have all the needed information to compute the z value:

```
z = (108-100) / 2.5 = 3.2
```


#### 7.5.4 Hypotheses 

Awesome! Now we can get the p-value from the z-value above. We see that it corresponds to `.9993`, but we have to remember to subtract this number from 1, making our p-value `0.0007`. Recall that a p-value below 0.05 is grounds for rejecting the null hypothesis. There, we do just that and conclude that there <i>is</i> an effect from the raw starch.


## 8.0 Correlation

Now, we'll look at relationships between variables. <b>Correlation</b> is a description of the relationship between two variables.

### 8.1 Covariance

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

### 8.2 Correlation

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

### 8.3 Confidence Intervals

The formal meaning of a confidence interval is that 95% of the confidence intervals should, in the long run, contain the true population parameter.

## 9.0 Mini Courses

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
