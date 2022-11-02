# Supervised Learning Classifier Midterm 
### by John Bae
## Background for The Gaussian Naive Bayes
  This algorithm combines the Naive Bayes methodology with a Gaussian distribution. 
  
  The Naive Bayes utilizes Bayes Theorem: $P(A | B) = \frac{P(B | A) * P(A)}{P(B)}$, where A is our Y values and B is the 
  individual columns of our X data respectively. The 'naivety' comes from the assumption that all of our seperate X columns 
  do not influence eachother, and as such we can treat them as 'independent events'. Generally we expect some correlation 
  between the various X datapoints, hence the naivety. I wanted to try it with this dataset to see how well it would classify 
  the skewed classes. There are several different distributions available to assign P(X) [here](https://scikit-learn.org/stable/modules/naive_bayes.html) .
  
  The Gaussian Distribution is a bell curve, the simplest form being $f(x) = exp(-x^2)$, a concave exponentiated quadratic. In our particular case we use the following: $P(x_i | y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} exp(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2})$ where $\sigma_y$ and $\mu_y$ are estimated from the training data. The only variable parameters in the scikit learn implementation are priors and var_smoothing. I only varied var_smoothing in my testing, as priors is utilized to feed the algorithm previously calculated class probabilities. I ended up with default parameters for this algorithm on my better test runs. I did not try partial_fit as I was not worried about memory overhead, however it may have helped the classifier. 

## Background for The Perceptron
  The Perceptron is an implementation of a generalized linear regression classifier. 
  
  The assumption with linear models is that $y(w, x) = w_0 + w_1x_1 +...+w_px_p$ where y is the predicted (encoded) class, x is the X data, and w is a vector of the coefficients, and y is the linear combination of the X data and w coefficients. In our more generalized model, we have a function h(X) such that $\hat{y}(w, X) = h(Xw)$. Additionally the W in our case is a matrix of the calculated coefficient weights. 
  
  The Perceptron is a simple classifier (found on [this](https://scikit-learn.org/stable/modules/linear_model.html#perceptron) list). It uses the same underlying algorithm as the [SDG](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier). This algorithm has many parameters that may be varied, of which I did testing on the Penalty, Alpha, fit_intercept, shuffle, n_iter_no_change, and class_weight parameters. Varying the penalty and alpha parameters had the largest effect on classification, as is to be expected as these values are fed directly to the mathematical linear model. 

## Background for The Support Vector Machine
  The Support Vector Machine is a set of multifunctional models.
  
  Support Vector Machines (SVMs) create a 'hyper-plane' (or set of them) that comes as close to linear seperation of the datapoints as possible. These Linear Seperation boundaries are created using the 'edge' datapoints, though since I ended up setting the kernel to 'poly' with a degree of 3 these lines take on a cubic shape. The constructed vectors comprising these deciscion boundaries are the 'Support Vectors' in the title of this algorithm. (The mathematics behind this algorithm is more advanced than I am able to describe effectively, but there is an excellent description of it found [here](https://scikit-learn.org/stable/modules/svm.html#mathematical-formulation)) 
  
  We make use of the Support Vector Classifier (SVC). There are many optional parameters for this Algorithm. These parameters had a much more noticeable effect on the classification results than the parameters for my other Algorithms. I tested variations of the C, kernel, degree, gamma, shrinking, and class_weight parameters. In particular, varying the kernel and C values led to better results. 
  
## Discussion of Models and Results
  Overall, the Support Vector Machine had the best results of the algorithms that I chose.
  
  I believe that the Gaussian Naive Bayes was a poor choice for this dataset, and that another function for the probabilities of the Naive Bayes might have achieved better results. The lack of variable parameters coupled with the fact that the classes did not seem to have a Gaussian Distribution (though it is possible I could have adjusted them to a Gaussian shape) led to some variance in the Accuracy Scores. I was able to get my results consistently over 0.80 accuracy, but I think another Naive Bayes would have been a better fit. 
  
  The Perceptron had some variation in results until I found an appropriate parameter set to feed into it. This may be because it is a linear function, or because of the way that the data was being scaled. Using the Robust scaler and the l1 penalty setting, I was able to get a relatively consistent accuracy greater than 0.83. 
  
  The Support Vector Machine performed the best both before and after finding the correct parameters. The best results were achieved with a Polynomial kernel of Degree 3, interestingly with no Regularization parameter. I thought that the C value would have a greater impact on the results. With my selected parameters, I was able to consistently achieve accuracy scores above 0.87. 
  
  If I were to try other methods, I would first learn about Pipeline construction and implement one. Additionally, I now have a better idea of how I would set up the ovverarching Class to be slightly more modifiable. I would attempt another Naive Bayes (one better fit to this situation), as well as possible a more complex Gaussian Process (after ensuring the class results agree with a Gaussian distribution). 
