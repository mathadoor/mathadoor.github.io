---
layout: post
title: Reflections on Bias-Variance Trade-off
date: 2023-10-15 11:12:00-0400
description: Reflections on bias-variance trade-off.
tags: Modeling Machine-Learning
categories: Reflections
giscus_comments: true
---


## Introduction
In a recent discussion with a fellow ML enthusiast, I realized my understanding of bias-variance tradeoff was unclear. I intuitively defended the stance that the tradeoff is no longer useful in understanding neural network, but I did not have a firm grounding to support the stance. In an effort to add some clarity, I am writing this article as a reflection on the applicability of bias-variance trade-off. The information presented is partly synthesized from my personal experience and another part from a collection of peer-reviewed publication. I hope to pursue some of the claims I made below in more detail and support with rigorous experiments in the future.  

## Parameter Estimation 
<!-- Explain what parameter estimation is -->
The whole idea of bias and variance comes into play from the exercise of estimating some hidden parameter. By hidden I mean we do not have direct access to it. Such a parameter can just be seen as a figment of imagination, but serves a crucial purpose in fulfilling some larger objective. To motivate the discussion, suppose I am a medical professional and I am interested in designing a diagnostic test for type-II diabetes. My understanding of bodily functions informs me the fasting blood glucose level of a diabetic patient may be higher than the ones without. Thus, I could potentially use this as a diagnostic test. So I put my head down and start designing my experiments with the objective of comparing  

<!-- Explain How parameters are estimated -->

<!-- How Bias-variance comes into play when estimating parameters -->

<!-- What is the limitation of this kind of bias-variance -->

<!-- Inductive Bias -->

<!-- Application of bias-variance trade-off to modern deep learning -->
In my previous [blog](https://mathadoor.github.io/blog/2023/modeling-basics/), I discussed the implication of statistical bias and variance. We had a biased coin and were interested in estimating the probability($$p_{head}$$) that it lands on a head. We were aware of the real value of $$p_{head}$$, estimated this value as $$\hat{p}_{head}$$ by sampling data and applying statistical methods to it. We referred to the sampled data as the training set and fitted our model to this set. The end goal of this exercise was to demonstrate the variation of the quality of fit based on the variation in the training set. The variation was performed along two dimensions:   

1) Resampling the training set for a fixed number of tosses. and   
2) Resampling the training set for a different number of tosses. 

As such, for a given number of tosses, one can evaluate the quality of fit based on the spread and the mean of $$\hat{p}_{head}$$ for different training sets. We referred to the former as the variance and the latter as the bias. As we increased the number of samples in the dataset, the spread reduced and the mean got closer to the actual value of $$p_{head}$$. Moreover, the variance and the bias was directly related to the capacity of the model. Higher the capacity, higher the variance and lower the bias. This is often the kind of picture of ML modeling painted in popular media. 

However, it is rarely the case that an ML practitioner can cleanly decompose the modeling exercise into its elemental constituents. In reality, there are a number of factors present that hinder this kind of analysis. Consequently, additional concepts must be introduced in our framework. In this article, we will take a look at what these hindering factors are. To limit the scope of this article, we will only deal with a few of them. Some of the factors that elicit a need for extending bias-variance trade-off are discussed in the subsequent section.

## Lack of Explicit Target Function

Machine Learning is inherently a function approximation problem. At an overarching level, ML problems can be defined in three different categories - supervised learning, unsupervised learning, and reinforcement learning. There are another set of learning paradigms, but they combine elements of the above. But in all of them, the central problem can reduced to that of function approximation. In supervised learning, the practitioner is given an input dataset and a set containing corresponding output values. They are tasked to learn a function estimating the relationship between input and output. In unsupervised learning, they are given a dataset and tasked to learn a function describing the underlying structure of the dataset. In reinforcement learning, they are tasked to estimate a function that guide an agent to take certain actions in an environment that maximizes its long term reward.   

Why does function approximation limit the applicability of bias-variance trade-off? Typically, the ground truth function is not explicitly available for comparison. As such, there is a need for using proxy methods to estimate the quality of the fit. This is typically done by splitting the training set in different parts, withholding one of them and estimating the performance on it. This practice emphasizes the quality of the signal in the dataset. Applying bias-variance trade-off to inadequate or poor signal in the training set will cause the practitioner to misunderstand the nature of the problem. For example, the collected data may imply that accuracy is maximized for linear regression when the actual underlying function is quadratic. Sample complexity is a useful concept as a remedy. The general idea behind sample complexity is to get an estimate of the number of samples needed to reach an accuracy within acceptable range. 

## Algorithmic Design Choices

Another issue with bias-variance trade-off is that it is a poor tool to reason about a model’s performance. For example, based on bias-variance discussion alone, it is not clear why convolutional neural networks should be used for image processing over fully connected neural networks. For this, the concept of inductive bias comes in handy. The concept is rooted in the fact that an ML algorithm typically considers a family of function, also known as hypothesis class to fit the data. During training, it selects the function that closely represents the target function during training. The propensity of the algorithm to consider this subset is referred to as inductive bias. 

It is important to note there a subtle difference between inductive bias and the statistical bias we mentioned previously. The statistical bias refers to the difference between the target function and the estimated function. On the other hand inductive bias refers to the family of functions considered by the algorithm. Thus it is independent of the target function. This separation between the target function and the hypothesis class lead to several benefits in designing a learning system. For example, the inductive bias can be used to characterize algorithms. The practitioner can then select the algorithms for further analysis based on the suitability of their inductive bias to capture the target function. 

Another benefit of inductive bias over statistical bias is it is tangible and intuitive. One can introduce various modeling constructs to induce a certain bias. For example, decision trees class overall considers a hypothesis class consisting of all possible decision trees. [ID3](https://en.wikipedia.org/wiki/ID3_algorithm) algorithm induces bias towards shorter trees by introducing [maximum information gain](https://www.section.io/engineering-education/entropy-information-gain-machine-learning/) principle. This type of inductive bias is also referred to as preference bias. The [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier) in Scikit-Learn also comes with a number of hyperparameters to restrict the class of trees considered for learning. For example, one can restrict the maximum depth of the trees by setting max_depth variable. This type of bias is known as restriction bias and it trims the original hypothesis class to a smaller class of hypothesis. Another beautiful example of inductive bias is that of convolutional layers along with the max pooling layer. They are specifically designed to induce a bias towards recognizing visual features than fully connected layers.

## Training Dynamics

The effect of solution evolution on learning cannot be explored with bias-variance trade-off. As noted earlier, ML algorithms start by considering a hypothesis class $$\mathcal{H}$$. It then subsequently shrinks this class towards a function that closely represents the target function to be captured. Inductive bias helps one understand the class of functions considered and preferred over the course of solution evolution. However, it only informs a general picture of the algorithm. The specific dynamics of the solution evolution as the model is fitted to a given data are unknown. For example, the empirical evidence suggests the learning rate schedule in neural network training can cause a dramatic effect on the neural network training. But the exact effect on the dynamics of the learning dynamics cannot be explored with the tools at hand. Luckily, in the context of neural network training, a powerful method called Neural Tangent Kernel is available. The method essential transforms the analysis from parameter space to a convex function space.

## Regimes Beyond Bias-Variance Trade-off

## References
[1] Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.  
[2] Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias–variance trade-off. Proceedings of the National Academy of Sciences, 116(32), 15849-15854.  
[3] Mitchell, T. M. (2007). Machine learning (Vol. 1). New York: McGraw-hill.  