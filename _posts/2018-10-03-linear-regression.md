---
layout: post
title: Linear Regression
subtitle: Linear Regression with one Variable
tags: [ml]
published: false
---

# Linear Regression with one Variable

Linear Regression is a class of supervised machine learning where we look for the answers that are real valued numbers and hence we call it **regression**.

## Predicting Housing Price

As a sample we take the task to predict the housing price of the houses in portland based on the size of the house.

We take this as a base sample data to understand the uni-variant linear regression problem. As we plot the price to size in a graph. It looks like the following.
![Housing Price Graph](../img/posts/.2018-10-03-linear-regression_images/2e881f1d.png)

Our goal is to develop a machine learning algorithm that can train itself to find a function that can draw a line as below so that, when a new data, say house with size as 1250 feet², using the line, we could easily predict the price to be around $220K.
![Predicted Housing Price](../img/posts/.2018-10-03-linear-regression_images/d3fc0ee1.png)

### Notations

To understand the notation, Lets take the pricing data as follow:


| Size in feet² (X)| Price ($) in 1000's(Y)|
| :------ |:--- |
|2104|460|
|1416|232|
|1534|315|
|852|178|
|...|...|

In the above table, each and every record is a training example that is fed to the system. The size of the house is the input to the machine learning system and it outputs the price as a real valued number. Hence, we denote the values in the following format.

> m - Number of training Examples <br>x - **input** variable / feature <br>y - **output** variable / **target** variable

To simplify things, (x,y) is denoted as one training example, whereas X(i),Y(i) represents ith training example.

So, in the above example, X(1)=2104 and Y(3)=315.

### Model

The System gets the training set, uses some algorithm to learn the hypothesis function. Using the hypothesis function (also called the model), we predict the price for any new given size of the house.
![Price Prediction](../img/posts/.2018-10-03-linear-regression_images\7b1d86bb.png)

Hypothesis(h) is the function that maps the input X to the output Y, such that `Y=h(X)`. 

The hypothesis can be represented as the following formula, for a uni-variant model
> h<sub>Θ</sub>(X) = Θ<sub>0</sub>+Θ<sub>1</sub>(X)
<br>where Θ<sub>0</sub> and Θ<sub>1</sub> are parameters of the model

So, For **Θ<sub>0</sub>=1.5** and **Θ<sub>1</sub>=0** the hypothesis looks like:
![Hypothesis Function](../img/posts/.2018-10-03-linear-regression_images\084a51aa.png)

And For **Θ<sub>0</sub>=0** and **Θ<sub>1</sub>=0.5** the hypothesis looks like:
![Hypothesis Function](../img/posts/.2018-10-03-linear-regression_images\717415b9.png)

So, For **Θ<sub>1</sub>=0** and **Θ<sub>1</sub>=0.5** the hypothesis looks like:
![Hypothesis Function](../img/posts/.2018-10-03-linear-regression_images\636e7675.png)

### Cost Function


> Disclaimer: Most part of the contents in this blog are from the [Machine Learning](https://www.coursera.org/learn/machine-learning) course by Andrew Ng.

