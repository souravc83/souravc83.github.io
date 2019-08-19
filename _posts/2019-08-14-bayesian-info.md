---
layout: post
title:  "Compare your Bayesian Models: BIC"
date:   2019-08-14 21:46:04
categories: statistics, bayesian
---

## Comparing Bayesian Models
 
 There a number of model selection methods in machine learning. In essence, the methods try to balance training errors with the complexity of the model. In this post, we want to look at the Bayesian Information Criterion. Although its final approximate form is used widely in practice, there is less emphasis on its actual theoretical basis. This is what we will try to address in this post.

Given different Bayesian models, how can we compare the different models with each other. As an example, think of the Gaussian mixture model. Let's say I am trying to fit the data I have with a mixture of K Gaussians. Different values of K correspond to different models. This is kinda equivalent to the question "how to choose K in K-means?" 

More formally, let's say we have a Bayesian model {% m %}M_i{% em %}, parameterized by the parameters {% m %}\theta_i{% em %}. We will start the analysis with a prior belief about {% m %}\theta_i{% em %} and then find update the belief after looking at the data. 

We might ask, what is the probability of the data we observe conditioned on this model. 
{% math %}p(D) = \int p(D,\theta_i)d\theta_i = \int p(D|\theta_i)p(\theta_i)d\theta_i{% endmath %} 

The Bayesian information criterion tries to maximize this probability. In plain English, it is saying that the best model is the one, that makes the data most likely. In other words, among all models, this looks like the model from which the data was drawn from most likely.

Now, this integral above is hard to evaluate. So, we will need to resort to an approximation. Let us first take a detour, and learn about Laplace Approximation.

## Laplace Approximation

Laplace approximation is a way of saying, if there is a complicated probability distribution we cannot find, we approximate it by a Gaussian, whose mean is at the mode of the distribution. The only other parameter we need to define is the variance. To define the variance, we will look at the Taylor expanstion of the function around its mode. We will show the result for the Taylor expansion for a scalar {% m %}z{% em %} and then we will show the extension when {% m %}z{% em %} is vector.


{% marginfigure 'mf-id-whatever' 'assets/img/test.png' 
'Laplace Approximation basically approximates a complicated probability distribution with a Gaussian centered at the mode of the distribution' %}

Now, lets see this mathematically. Lets say there is a probability density function 
{% math %}p(z) = \frac{1}{Z}f(z){% endmath %} 
where {% m %}Z = \int f(z)dz{% em %}, which as we know, is in general, a difficult integral to compute. But since we have decided to approximate this with a Gaussian, and we know a Gaussian is a valid probability distribution, the Laplace approximation provides us an easy way to approximate this integral.

Now, lets consider a point {% m %}z_0{% em %} which is the mode. Therefore, by definition, {% m %}f'(z_0)= 0{% em %}. If we expand {% m %}ln(f(z)){% em %} in a taylor expansion around {% m %}z_0{% em %}, we get:
{% math %}ln(f(z)) = ln(f(z_0)) + \frac{d}{dz}ln(f(z))|_{z_0}(z-z_0) + \frac{d^2}{d^2z}ln(f(z)|_{z_0}(z-z_0)^2 + ...{% endmath %} 

Notice that the second term will be zero, because 
{% math %}\frac{d}{dz} ln(f(z))|_{z_0} = \frac{1}{f(z_0)} f'(z_0) = 0{% endmath %} 

Therefore, we can write
{% math %}ln(f(z)) = ln(f(z_0)) - \frac{1}{2}A(z-z_0)^2{% endmath %} 

Here {% math %}A = - \frac{d^2}{dz^2} ln(f(z))|_{z_0}{% endmath %} 

Taking exponentials
{% math %}f(z) = f(z_0)exp(-\frac{A }{2}(z - z_0)^2){% endmath %} 

Now, recognize that if this were a Gaussian, then {% m %}A = \frac{1}{\sigma^2}{% em %}. 


Now, in the multivariate case, you can go through a similar derivation. The only difference is, {% m %}A{% em %} will be the Hessian matrix, instead of just being a single second derivative. This will correspond to the precision matrix of the multivariate normal, i.e. the inverse of the covariance matrix. If {% m %}z{% em %} is a vector, we can approximate the function {% m %}f(z){% em %} as a multivariate normal, around its mode

{% math %}f(z) =  f(z_0)exp(-\frac{1}{2}(z-z_0)^T A (z-z_0)){% endmath %} 

What this tells us, is that by ignoring the higher order terms of the Taylor expansion, we have managed to express an arbitrary function f(z) near its mode with a normal distribution, whose mean is the mode of the function, and precision matrix is the Hessian at {% m %}z_0{% em %}. 

Repeating again what we said before,  {% m %}Z = \int f(z)dz{% em %} is difficult to calculate. Now, lets use the Laplace approximation to get it.
{% math %}
\begin{align}
Z = \int f(z)dz = f(z_0)exp(-\frac{1}{2}(z-z_0)^T A (z-z_0)) dz \\ = f(z_0) \frac{(2\pi)^{M/2}}{|A|^{1/2}}\int \frac{|A|^{1/2}}{(2\pi)^{M/2}}exp(-\frac{1}{2}(z-z_0)^T A (z-z_0)) dz = f(z_0) \frac{(2\pi)^{M/2}}{|A|^{1/2}}
\end{align}{% endmath %} 

So, what happened here? In the first step, we wrote the expression we have derived before, for {% m %}f(z){% em %}. Then we multiplied and divided the right hand side by {% m %}\frac{(2\pi)^{M/2}}{|A|^{1/2}}{% em %} and moved the integral to contain terms with z. Now, we multiplied and divided to make everything inside the integral sign equal to the density of the normal distribution, in M dimensions (A is the precision, i.e. the inverse of the covariance matrix). The density integrates to 1, and hence we are left with the term outside the integral.

In practice what this meant is that, to find this complicated integral, all that we need to do is to find the value of the function at {% m %}z_0{% em %} where the function reaches a maximum, and also the Hessian matrix A at {% m %}z_0{% em %}. Then we are all set to calculate the complicated integral we are after.


## Back to Bayesian Information Criterion

So, where we got stuck last time was that we did not know how to evaluate the integral to get the marginal. We wanted to evaluate: 
{% math %}p(D) = \int p(D,\theta_i)d\theta_i = \int p(D|\theta_i)p(\theta_i)d\theta_i{% endmath %} 

Here {% m %}f(\theta) = p(D|\theta)p(\theta){% em %} and therefore, using Laplace approximation, 

{% math %}p(D) = p(D|\theta_{MAP})p(\theta_{MAP})\frac{(2\pi)^{M/2}}{|A|^{1/2}}{% endmath %} 

Taking log
{% math %}ln(p(D)) = ln(p(D|\theta_{MAP}) + ln(p(\theta_{MAP})) + \frac{M}{2} ln(2\pi) - \frac{1}{2} ln|A|{% endmath %} 

Now, let's assume that N is large, and terms that depend on N dominate this expression. 
It is easy to see that {% m %}ln(p(D|\theta_{MAP}) {% em %} grows with the size of the data. For example, if the data is IID, then 
{% math %}ln(p(D|\theta_{MAP}) = \Sigma_{i=1}^n ln(p(D_i|\theta_{MAP}){% endmath %} 

Similar to the likelihood, the hessian of the likelihood grows with the size of the data. Now, we will assume that the Hessian linearly increases with N, i.e. {% m %}A \approx N A_0{% em %}.
So, therefore, from the properties of determinants
{% math %}|NA_0| = N^M A_0{% endmath %} 
if {% m %}A_0{% em %} is an M dimensional matrix. This is because imagine each term in the determinant is a product of m terms, each multiplied by N (in a 2D matrix [[a,b], [c,d]], the determinant is (ad - bc)). 

We can ignore the terms that do not grow with the size of the data. These are the terms {% m %}ln(p(\theta_{MAP})){% em %} and {% m %}\frac{M}{2} ln(2\pi){% em %}. 

Hence, we can now write:
{% math %}
\begin{align}
ln(p(D)) \approx ln(p(D|\theta_{MAP}) - \frac{1}{2} ln|A|\\  =  ln(p(D|\theta_{MAP}) -   \frac{1}{2} lnN^M|A_0| \\ \approx ln(p(D|\theta_{MAP}) - \frac{M}{2} lnN
\end{align}
{% endmath %} 

This result defines the **Bayesian information criterion**. The Bayesian information criterion is important because it scores a model higher if the likelihood of the data given the posterior mode is high, but it penalizes a model that uses a large number of parameters(in this case M). 
