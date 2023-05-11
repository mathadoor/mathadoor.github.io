---
layout: post
title: Modeling Basics and Statistical Bias
date: 2023-05-10 11:12:00-0400
description: A study of statistical bias for modeling a simple prediction problem. 
tags: Theory Probability Modeling
categories: Fundamentals
giscus_comments: true
---
*"All models are wrong, but some are useful” - George Box*

## Introduction

Following our discussion on mind-projection fallacy in my previous article, I want to now talk about what happens when we model a simple scenario. In this article, I take you through a fictitious example of modelling a prediction problem. We are given a coin and we are tasked to estimate the probability that it will fall on heads upon tossing. We have no idea how this coin behaves, but we might discover something advantageous by playing with it. We look to statistical learning methods to guide our discovery. 

In the next section, I discuss what kind of models are appropriate to estimate the probability. To this end, we discuss two simple models with different capacity. The subsequent section presents the experiments I performed to contrast these models. I consider two different data generating processes as an approximation of the game to collect the data, which is use later to fit the models. I then present the results of these experiments, followed by a discussion contrasting the applicability of these models and our findings.

## Methodology

First and foremost, we need a model to represent the behaviour of the coin. More specifically, we are interested in capturing certain aspects of the data-generating process. You may recall that the process of repeated coin toss is also known as Bernoulli trials. Here, we assume each coin toss is independent of the other and the coin has a probability p with the heads as the outcome of a toss. Thus, we can estimate p by performing a number of trials and computing the value that maximizes the probability. Before we go further, we need to emphasize the concept of probability here. The probability here represents a degree of plausibility that we measure in the range of 0 and 1. It is important to note this degree of plausibility is entirely a mathematical tool and may not represent reality itself. After all, the outcome of a coin toss is either heads or tails. Perhaps, we could accurately predict the exact outcome of the toss by running a multi-physics simulation that models the dynamics of the coin toss in the presence of air resistance and gravity. However, these efforts might be too complex to realize. We instead start with a simpler model based on our intuition of the data-generating process. 

Now, we start by generating samples. Suppose we perform $$N$$ trials out of which we get m heads and n tails. It can be shown the fraction of trial for which the coin fell on heads is the maximum likelihood estimated(MLE) value of $$p$$. In layman’s terms, this is the value of $$p$$ that maximizes the plausibility of observing the outcomes we did given the data model. To proceed further, let us first go through how the above computation unfolds to make an MLE of $$p$$. We represent the outcome of trial $$i$$ as $$X_i$$. The probability that we observe such outcomes condition on the value of $$p$$  is:

$$
P(X_1, X_2, .. X_n|p) = \prod_i^n P(X_i|p)
$$

Notice the right-hand side decomposes the conditional events. This is valid under the assumption the trials are independently distributed as we noted previously. Now expanding the right-hand side part of the equation is easy:

$$
\prod_i^n P(X_i|p) = p^m(1-p)^{n-m}
$$

$$
\cfrac{\partial P(X_1, X_2, .. X_n|p)}{\partial p} \Biggm\lvert_{p_{MLE}}= \cfrac{\partial p^m(1-p)^{n-m}}{\partial p} \Biggm\lvert_{p_{MLE}}= 0
$$

$$
\Rightarrow p_{MLE} = \cfrac{m}{m+n}
$$

It is customary to consider several models for comparison. To this end, we consider a slight variation of the Bernoulli trials. We lift the condition of independence between trials. We chose a model with Markov property between subsequent trials. If the coins is facing heads up, it is likely to come up with heads by probability $$p + \delta$$, otherwise it is $$p - \delta$$. The figure below illustrates the difference between the state diagrams of the model without and with Markov property.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/modelingbasics/state_diagram.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. State Diagram of the Bernoulli trials with and without Markov property.
</div>

We are now required to estimate both the parameters in case of the model with Markov property. Note, the one with Markov property is generalization of the one without. By setting $\delta=0$, we recover Bernoulli trial. Now, just like in the case of Bernoulli trials, we wish to estimate the maximum likelihood values of both $$\delta$$ and $$p$$. The likelihood is formulated as follows:

$$
P(X_1, X_2, .. X_n|p) = P(X_o|p, \delta)\prod_i^n P(X_i|p, X_{i-1})
$$

We divide our sequence of trials into pairs of subsequent trials to reduce the above formulation further. We can have four types of such pairs - $$HH$$, $$HT$$, $$TH$$, and $$TT$$. Suppose we have $$m$$, $$n$$, $$r$$ and $$s$$ number of occurrences of such pairs. Then the above formulation reduces to the following expression:

$$
P(X_1, X_2, .. X_n|p) = P(X_o|p, \delta) (p \ + \delta)^m(p \ - \delta)^r(1 - (p \ + \delta))^n(1 - (p \ - \delta))^s
$$

We can then compute the formula for the MLE values using calculus just like we did before. Given the complexity of the model, we will estimate these values numerically instead.

## Experiments

We perform four experiments in total. We generate two types of datasets assuming Bernoulli trials without Markov property and with Markov property. We simulated these datasets by assuming $$p = 0.7$$ and $$\delta = 0.1$$. These values are chosen arbitrarily so that the coin is biased in both the cases and the data generated with Markov property has a more complex process than the without. We vary the number of samples per dataset from 10 to 100 in steps of 10. For each setting, we generate 100 datasets to accurately compute the mean values of $$p_{MLE}$$ and $$\delta_{MLE}$$.

## Results and Discussion

The results of our experiments are plotted in the figures below. Figure 2 presents $$p_{MLE}$$ computed for the data generated with Bernoulli coin. Dashed line represents the true value of p, solid lines represent the mean values of p estimated by Bernoulli and Markov model, and the shaded region represents the standard deviation for both the models. Notice both models are able to accurately estimate the true value of p as indicated by the proximity of the mean value to the true value. However, the estimation by the Markov model has higher variance than the Bernoulli model. This is because the Bernoulli model has fewer excess parameters to capture the underlying data generation process. On the other hand, Markov model has the parameter $$\delta$$ which may overfit to the superficial irregularities causing the estimated p to have higher variance. Figure 4 further corroborate our claim. Notice the model fits a non-zero value to $$\delta$$ for all experiments.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0" style="text-align: center;"  text-align=center>
        {% include figure.html path="assets/img/modelingbasics/bernoulli_data.png"  %}
    </div>
</div>
<div class="caption">
    Figure 2. Maximum Likelihood Estimate of p for Bernoulli Coin with different models
</div>
Figure 3 presents the results of fitting our models to the data generated for the Markov coin. Since Bernoulli model assumes $$\delta = 0$$, it fails to account for the Markov property and ends up assigning a higher value to $$p$$ than its true value. Markov model on the other hand is able to capture the true value of $$p$$ accounting for the Markov property. We can see in Figure 4 that the model accurately predicts the value of $$\delta$$.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0" style="text-align: center;" text-align=center>
        {% include figure.html path="assets/img/modelingbasics/markov_data.png"  %}
    </div>
</div>
<div class="caption">
    Figure 3. Maximum Likelihood Estimate of p for Markov Coin with different models
</div>

Finally, note the spread in all figures reduce as we increase the number of samples per dataset. This is aligned with our expectations as larger dataset allows the model to capture more signal, which in turn increases its precision.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0" style="text-align: center;" text-align=center>
        {% include figure.html path="assets/img/modelingbasics/delta.png"  %}
    </div>
</div>
<div class="caption">
    Figure 4. Maximum Likelihood Estimate of delta for both coins with Markov Model
</div>

## Conclusion

In this article, we studied the basics of modelling by considering two toy datasets. The datasets are generated by a biased coin with and without Markov property. We fitted two models with varying capacities to the datasets generated for these processes. We chose varying capacities to demonstrate the sample complexity of training them. The training objective is to estimate the probability of getting a head for a coin toss. We found Bernoulli Model(the one with lower capacity) requires fewer samples than Markov model(more capacious) to precisely estimate the probability of the Bernoulli coin. Both models are equally accurate. However, as we increase the number of samples per dataset, both models become more precise. On the other hand, the higher capacity of the Markov model allows it to accurately estimate the probability of the Markov coin, while Bernoulli model fails to capture the effect of the previous state. We argue the difference in this performance is because Markov model has more capacity thus it can overfit to superficial irregularities. It requires more samples to ignore them. On the other hand, simpler models like Bernoulli can easily ignore them as they assume their non-existence. However, models can be inaccurate if it fails to account for some aspects of the data generation process. This behaviour is clearly seen for the Markov coin. 

This phenomenon of how the alignment between the model and the data generating process affect the model accuracy and precision is a classic case of statistical bias-variance trade-off[1]. It is important to note the discussion on bias-variance trade-off in popular machine learning literature on the web attributes the difference in the prediction and the ground truth to the capacity of the model. However, this argument is incomplete as we must take the data generating process into account. A more capacious model does not have to exhibit low bias. Also, modern machine learning theory tells us more capacious model can increase both the accuracy and the precision[2]. Finally, statistical bias is just a part of the story. There are other forms of bias present in machine learning algorithms[3] that I intend to cover in one of the subsequent articles.

## References
[1] Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.  
[2] Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias–variance trade-off. Proceedings of the National Academy of Sciences, 116(32), 15849-15854.  
[3] Mitchell, T. M. (2007). Machine learning (Vol. 1). New York: McGraw-hill.  