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
> h<sub> \theta Θ</sub>(X) = Θ<sub>0</sub>+Θ<sub>1</sub>(X)
<br>where Θ<sub>0</sub> and Θ<sub>1</sub> are parameters of the model

So, For **Θ<sub>0</sub>=1.5** and **Θ<sub>1</sub>=0** the hypothesis looks like:
![Hypothesis Function](../img/posts/.2018-10-03-linear-regression_images\084a51aa.png)

And For **Θ<sub>0</sub>=0** and **Θ<sub>1</sub>=0.5** the hypothesis looks like:
![Hypothesis Function](../img/posts/.2018-10-03-linear-regression_images\717415b9.png)

So, For **Θ<sub>1</sub>=0** and **Θ<sub>1</sub>=0.5** the hypothesis looks like:
![Hypothesis Function](../img/posts/.2018-10-03-linear-regression_images\636e7675.png)


### Cost Function

Now, lets bring the graph and training examples together.
![Hypothesis Function with training examples](../img/posts/.2018-10-03-linear-regression_images\56c08adc.png)

Here, the hypothesis function is the line for a given  Θ<sub>0</sub> and  Θ<sub>1</sub>. And the X denotes the training data for say, the housing price.

We need the hypothesis function to be very close to value of Y, so that we can use this hypothesis function to calculate the output for any new input data. In other words, we need to choose Θ<sub>0</sub>,Θ<sub>1</sub> so that h(X) is very close to Y. 

To rephrase it to a mathematical notation, our goal is to minimize the value of the difference between hypothesis and output value of the training data.

So, we are trying to minimise the squared difference between the hypothesis and actual value and averange across all the **m** training examples. We then half the difference to make the number smaller for caluclation.

The overall goal is

> <img src="http://www.sciweavers.org/tex2img.php?eq=min_%7B%20%5Ctheta%20_%7B0%7D%2C%5Ctheta%20_%7B1%7D%7D%20%20%5Cfrac%7B1%7D%7B2m%7D%20%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%28h%28x%5E%7B%28i%29%7D%29-y%5E%7B%28i%29%7D%29%5E2&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="min_{ \theta _{0},\theta _{1}}  \frac{1}{2m}  \sum_{i=1}^{m}(h(x^{(i)})-y^{(i)})^2" width="246" height="50" />

Here we are trying to find the squared difference because it is the most commonly used method that works reasonably well than most other cost functions in a wide variety of applications.

Now, we specify the cost function as 
> <img src="http://www.sciweavers.org/tex2img.php?eq=J%28%5Ctheta%20_%7B0%7D%2C%5Ctheta%20_%7B1%7D%29%20%3D%20%5Cfrac%7B1%7D%7B2m%7D%20%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%28h%28x%5E%7B%28i%29%7D%29-y%5E%7B%28i%29%7D%29%5E2&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="J(\theta _{0},\theta _{1}) = \frac{1}{2m}  \sum_{i=1}^{m}(h(x^{(i)})-y^{(i)})^2" width="268" height="50" />
And our overall goal is to minimize this cost function:

> <img src="http://www.sciweavers.org/tex2img.php?eq=min_%7B%5Ctheta%20_%7B0%7D%2C%5Ctheta%20_%7B1%7D%7D%20%26%20J%28%5Ctheta%20_%7B0%7D%2C%5Ctheta%20_%7B1%7D%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="min_{\theta _{0},\theta _{1}} & J(\theta _{0},\theta _{1})" width="151" height="19" />

#### Intuition

To grow a better intuition about what cost function is and what the minimisation of cost function does, we consider Θ<sub>0</sub> to be 0.

So, our overall goal is 

> min<sub>Θ<sub>1</sub></sub> J(Θ<sub>1</sub>) where <img src="http://www.sciweavers.org/tex2img.php?eq=J_%7B%5Ctheta%20_%7B1%7D%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20%28h_%7B%5Ctheta%20_%7B1%7D%7D%28X%5E%7B%28i%29%7D%29-Y%5E%7B%28i%29%7D%29%5E2&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="J_{\theta _{1}} = \sum_{i=1}^{m} (h_{\theta _{1}}(X^{(i)})-Y^{(i)})^2" width="207" height="50" />

> Disclaimer: Most part of the contents in this blog are from the [Machine Learning](https://www.coursera.org/learn/machine-learning) course by Andrew Ng.

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTk0MjMyMTQxXX0=
-->