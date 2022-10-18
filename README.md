# Supervised Learning Classifier Midterm
## Background for The Gaussian Naive Bayes
  This algorithm combines the Naive Bayes methodology with a Gaussian distribution. 
  
  The Naive Bayes utilizes Bayes Theorem: $P(A | B) = \frac{P(B | A)*P(A)}{P(B)}$, where A is our Y values and B is the 
  individual columns of our X data respectively. The 'naivety' comes from the assumption that all of our seperate X columns 
  do not influence eachother, and as such we can treat them as 'independent events'. Generally we expect some correlation 
  between the various X datapoints, hence the naivety. I wanted to try it with this dataset to see how well it would classify 
  the skewed classes. There are several different distributions available to assign P(X) [here](https://scikit-
  learn.org/stable/modules/naive_bayes.html).
  
  The Gaussian Distribution is a bell curve, the simplest form being $f(x) = exp(-x^2)$, a concave exponentiated quadratic. In our particular case we use the following: $P(x_i | y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} exp(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2})$ where $\sigma_y$ and $\mu_y$ are estimated from the training data. The only variable parameters in the scikit learn implementation are priors and var_smoothing. I only varied var_smoothing in my testing, as priors is utilized to feed the algorithm previously calculated class probabilities. I ended up with default parameters for this algorithm on my better test runs. I did not try partial_fit as I was not worried about memory overhead, however it may have helped the classifier. 

## Background for The Perceptron
  The Perceptron is an implementation of a linear regression classifier. 
  
  The assumption 


## Background for The Support Vector Machine
