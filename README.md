# Practical data analytics guide:  Finding explicit polynomial relations in problems with feature interactions and breakpoints
### **Problem statement:**

When studying real-world engineering data, one usually wants to find models that makes sense from a physical point of view, and identify the parameters of the model through data analysis. However, in complex systems (for example electric machines or fluid mechanics), several physical phenomena compete with each other, and complex interactions may occur. 

For illustration purposes, let's consider the following problem:
We have a (fake) dataset of five features  X1, X2, X3, X4  and   X5  and a target value  Y 

The ground-truth model (which is supposed to be unknown) is the following (*note that Y does not depend on X5* : the model should be able to detect this):

 Y = (4*X_1^2 + 2*X_1*X_2 + 0.005*X_2^3)   for   X_4 < 200  

 Y = (2*X_3^2 + 10*X_2*X_3 + 0.01*X_1^3)   for   X_4 < 400  

 Y = (3*X_3^2 + 5*X_2*X_3 + 0.02*X_1^3)   for   X_4 < 700  

 Y = (X_1*X_3 + 4*X_1*X_2 + 0.002 * X_2^3)    for   700 =< X_4  

In this study, data is generated for X1, X2, X3, X4, X5 and Y is computed with the above model. Gaussian noise is then added around Y in order to be closer to what's happening in reality.

This model combines breakpoints (with X_4) and multi parameter regressions of different degrees. The question is:
**Is it possible to find explicitly this ground-truth model through data analysis and ML methods ?**

**Why it is not that simple:**

Plotting   Y   against   X_i   or looking at correlations will not be sufficient to grasp the underlying relationships. Moreover, off-the-shelf machine learning algorithms may easily work to make multi-parameters polynomial regressions, or make decision trees that can handle non-polynomial relationship (for instance, dependence on a breakpoint, as in the model above). However, what we are looking for is a **combination** of piecewise polynomial regressions and decision trees.

The **first part** of this study will be a simple case with just one feature  X  and one target value  Y , with  Y = f(X)  where **f is a piecewise polynomial function**. We will use the very useful package [pwlf](https://jekel.me/piecewise_linear_fit_py/index.html) made by Charles Jekel.

The **second part** will be about the **multi-parameter polynomial regression** case (without breakpoints of any kind).

The **third part** will finally tackle the **multi-parameter polynomial regression with breakpoints**.

***Some additionnal comments:***
- In this notebook, some graphs are interactive (generated with plotly), others are not, in order to avoid slowness.
- For the sake of simplicity, we won't be performing standard preprocessing such as scaling, even though it would probably improve performance in some cases. 

## TL;DR (Summary):
To tackle this problem, two possible pathways:
- 1/ **Look systematically for multilinear regressions with polynom and interaction features** (for instance, for two features  X1  and  X2 , look for  X1^2  ,  X1X2 ,  X2^2 ,  X1^3  ...). Scikit learn has efficient built-in methods for this. One challenge can be to identify the right number of parameters / polynom degrees (the usual problem of underfitting vs overfitting). One other challenge is that interactions can be non polynomial (for instance if there are breakpoints: think about the Reynolds number in fluid mechanics). Packages such as **pwlf** can help identify breakpoints in a model.
- 2/ Train a ML model such as XGBoost, regression tree, a neural network and use **explainability methods** to extract the underlying behaviour of the model. In this notebook, we use SHAP (SHapley Additive exPlanations) which is, in 2022, a widely used package for this purpose.

These two methods have their own benefits and drawbacks, and Machine Learning / data analytics is about trying solutions, combining them and iterating. Some problems are more adated to the first method, some to the second, some to a combination of the two.
