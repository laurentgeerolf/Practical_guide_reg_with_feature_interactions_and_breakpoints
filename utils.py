#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pwlf
from kneed import KneeLocator
from sympy import Symbol
from sympy.utilities import lambdify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error


def piecewise_lin_reg(X, Y, nb_of_lines = 5, degree = 1):
    """
    Uses the pwlf package to compute a piecewise linear regression of Y = f(X)
    Returns the fitted model
    """
    my_pwlf = pwlf.PiecewiseLinFit(X, Y, degree=degree)
    my_pwlf.fit(nb_of_lines)
   
    return my_pwlf

def find_relationship(X,Y,nb_of_lines = range(2,10), degree = 1):
    """
    Computes different piecewise regressions with nb_of_lines as variable.
    Returns the breakpoints, r2 and standard errors (se) in the result dataframe
    """
    results = pd.DataFrame()
    for n in nb_of_lines:
        print(f"Computing with {n} lines ...")
        my_pwlf = piecewise_lin_reg(X,Y,n, degree = degree)
        results.loc[n,"nb_lines"] = n
        results.loc[n,"breakpoints"] = "; ".join([str(round(c,1)) for c in my_pwlf.fit_breaks.tolist()[1:-1]])
        results.loc[n,"r2"] = my_pwlf.r_squared()
        results.loc[n,"se"] = "; ".join([str(round(c,2)) for c in my_pwlf.standard_errors().tolist()])
        
    return(results)

def find_optimal_regression(results, column_X = "nb_lines", column_Y = "r2"):
    """
        Finds the optimal number of lines kn (as determined by the "elbow value" of Y). Uses the kneed package.
    """
    
    kn = KneeLocator(results[column_X], results[column_Y], curve='concave', direction='increasing').knee
    if not kn:
        print(f"Knee not found: taking maximum value of {column_Y}")
        kn = results.reset_index(drop=True).loc[results[column_Y].argmax(),column_X]
    return kn


def plot_the_results(results):
    """
    Plots the "result" dataframe, computed with the find_relationship function
    """
    to_plot = pd.melt(results["breakpoints"].str.split("; ", expand = True).reset_index().astype(float), id_vars= "index").dropna().drop(columns = ["variable"])

    kn = find_optimal_regression(results)
    
    print(f"Knee found: {kn} lines")
    
    fig = make_subplots(rows=2, cols=2,
                    row_heights=[0.2, 0.8],
                    vertical_spacing = 0.02,
                    shared_yaxes=False,
                    shared_xaxes=False,
                    specs=[[{}, {"rowspan": 2}],
                   [{}, {}]],
                    subplot_titles=("Fig 1 : Distribution of breakpoints (X) for different number of lines (Y)",
                                    "Fig 2 : R2 values"))

    fig.add_trace(go.Histogram(x=to_plot['value'], name = 'Histogram', orientation = "v",
                              xbins = dict(size = (to_plot['value'].max()-to_plot['value'].min())/20)), row = 1, col = 1)

    fig.add_trace(go.Scatter(x=to_plot['value'], y=to_plot['index'], name = 'Breakpoint',mode='markers' ), row = 2, col = 1)
    fig.update_layout(template = "seaborn")
    fig.add_trace(go.Scatter(x = results["nb_lines"], y = results["r2"], mode = "lines", name = "r2" ), row = 1, col = 2)
    fig.add_trace(go.Scatter(x = [kn], y = results.loc[results.nb_lines==kn,"r2"], mode = "markers", name = "Elbow value" ), row = 1, col = 2)
    fig.add_hrect(y0= kn - 0.5, y1=kn + 0.5, row=2, col=1,
              annotation_text="best fit", annotation_position="top left",
              fillcolor="green", opacity=0.25, line_width=0)
    fig.update_annotations(font_size=12)
    fig.update_yaxes(title_text="Nb of lines", row=2, col=1)
    fig.update_yaxes(title_text="R2", row=1, col=2)
    return fig


x = Symbol('x')

def get_symbolic_eqn(pwlf_, segment_number):
    if pwlf_.degree < 1:
        raise ValueError('Degree must be at least 1')
    if segment_number < 1 or segment_number > pwlf_.n_segments:
        raise ValueError('segment_number not possible')
    # assemble degree = 1 first
    for line in range(segment_number):
        if line == 0:
            my_eqn = pwlf_.beta[0] + (pwlf_.beta[1])*(x-pwlf_.fit_breaks[0])
        else:
            my_eqn += (pwlf_.beta[line+1])*(x-pwlf_.fit_breaks[line])
    # assemble all other degrees
    if pwlf_.degree > 1:
        for k in range(2, pwlf_.degree + 1):
            for line in range(segment_number):
                beta_index = pwlf_.n_segments*(k-1) + line + 1
                my_eqn += (pwlf_.beta[beta_index])*(x-pwlf_.fit_breaks[line])**k
    return my_eqn.simplify()


def print_equation(my_pwlf_2):
    eqn_list = []
    f_list = []
    for i in range(my_pwlf_2.n_segments):
        eqn_list.append(get_symbolic_eqn(my_pwlf_2, i + 1))
        print(f"Equation number: {i + 1} (for {round(my_pwlf_2.fit_breaks[i],2)}< x < {round(my_pwlf_2.fit_breaks[i+1],2)})" )
        print(f"{eqn_list[-1]}")
        f_list.append(lambdify(x, eqn_list[-1]))
        
        
def find_best_model(X,Y,Y_ref = None,nb_of_lines = range(2,13), display_plot = True, degree = 1):
    X_train, X_val, y_train, y_val = train_test_split(X,Y,random_state=0, test_size=0.2)
    regression_results = find_relationship(X_train,y_train,nb_of_lines = nb_of_lines, degree = degree)
    kn = find_optimal_regression(regression_results)
    fig1 = plot_the_results(regression_results)
    #Choice of final regression
    final_lin_reg = piecewise_lin_reg(X_train, y_train, nb_of_lines = kn, degree = degree)
    Y_pred = final_lin_reg.predict(X)
    Y_pred_val = final_lin_reg.predict(X_val)   
    
    if display_plot:
        fig1.show()
    print("Final Model :")
    print_equation(final_lin_reg)
    print(f"The RMSE of the final model on the test set is: {round(np.sqrt(mean_squared_error(Y_pred_val, y_val)),2)}")

    if display_plot:
            #plotting final result
#         in plotly
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=X, y=Y,
#                             mode='markers',
#                             name='Real data'))
#         if type(Y_ref) != type(None):
#             fig.add_trace(go.Scatter(x=X, y=Y_ref,
#                                 mode='markers',
#                                 name='Ground-truth model'))
#         fig.add_trace(go.Scatter(x=X, y=Y_pred,
#                             mode='markers',
#                             name='Computed model'))
#         fig.update_layout(template = "seaborn")
#         plt.figure()
#         fig.update_layout(title = "Fig 3: Final model")
#         fig.show()

    #in seaborn
        plt.figure()
        sns.scatterplot(x= X, y = Y, label = "Real data")
        sns.lineplot(x= X, y = Y_ref, label = "Ground-truth model", color = "red")
        sns.lineplot(x= X, y = Y_pred, label = "Computed model", color = "green")
        plt.title("Fig 3: Final model")
        plt.show()

    
    return (final_lin_reg, regression_results)



def polynomial_transform(df_in, max_degree = 3):
    poly = PolynomialFeatures(max_degree)
    poly.fit(df_in)
    columns = poly.get_feature_names_out()
    df_out = pd.DataFrame(poly.transform(df_in))
    df_out.columns = columns
    return df_out

def multi_linear_regression(df, target, display_plot = True):
    X = df.values
    Y = target
    X_train, X_val, y_train, y_val = train_test_split(X,Y,random_state=0)
    model = LinearRegression().fit(X_train, y_train)
    perm_importance = permutation_importance(model, X_val, y_val,
                           n_repeats=30,
                           random_state=0)
    if display_plot:
        to_bar = pd.DataFrame(perm_importance.importances_mean, index = [str(f) for f in df.columns]).sort_values(by = 0, ascending = True)
        plt.figure()
        plt.barh(to_bar.index, to_bar[0])
        plt.xlabel("Permutation Importance")
        plt.show()
    return perm_importance.importances_mean, df.columns, model

def n_best_features(perm_importance, column_names, n = 5):
    return(column_names[np.argpartition(perm_importance, -n)[-n:]])


def print_model(coeff, columns, nb_decimals = 5):
    out = ""
    for i in range(len(coeff)):
        if i == 0:
            out += "Model found : " + str(round(coeff[i], nb_decimals)) + "*" + columns[i]
        else:
            out += " + " + str(round(coeff[i], nb_decimals)) + "*" + columns[i]
    return(out)


def find_best_polynomial_model(df, Y, max_degree = 3, max_parameters = 5, nb_decimals = 5, display_plot = True):
    
    #First regression, to determine the most relevant features
    pol_transform = polynomial_transform(df, max_degree)
    perm_importance, column_names, _ = multi_linear_regression(pol_transform, Y, display_plot)
    final_features = n_best_features(perm_importance, column_names, n = max_parameters)

    #Final regression, to determine the model with selected features
    _, _, model = multi_linear_regression(pol_transform[final_features], Y, display_plot)
    print(print_model(model.coef_, final_features, nb_decimals = nb_decimals))
    return model, final_features


def plot_pol_reg_results(rmse_results, log = True):
    to_plot = rmse_results.copy()
    if log:  
        to_plot[["training_error","test_error"]] = np.log10(to_plot[["training_error","test_error"]])
    else:
        to_plot[["training_error","test_error"]] = to_plot[["training_error","test_error"]]
    g = sns.FacetGrid(to_plot, col="degree")
    g.map(sns.lineplot, "nb_parameters", "training_error", alpha=.7, label = "training error")
    g.map(sns.lineplot, "nb_parameters", "test_error", alpha=.7, color = "r", label = "test error")
    g.add_legend()
    g.fig.subplots_adjust(top=0.8)
    if log:
        g.fig.suptitle('Log10 of training and test errors for different hypothesis of max_degree and number of parameters')
    else:
        g.fig.suptitle('Training and test errors for different hypothesis of max_degree and number of parameters')
    return g

def make_polynomial_regressions(X,Y,max_degree = 3, parameter_range = range(3,8)):
    X_train, X_val, y_train, y_val = train_test_split(X,Y,random_state=0)
    
    rmse_results = pd.DataFrame()
    i=0
    for degree in range(2,max_degree + 1):
        for max_parameters in parameter_range:
            print(f"Finding model with {max_parameters} parameters and degree = {degree}")
            model, features = find_best_polynomial_model(X, Y, max_degree = degree, max_parameters = max_parameters, display_plot = False)
            print("-"*50)
            rmse_results.loc[i, "degree"] = degree
            rmse_results.loc[i, "nb_parameters"] = max_parameters
            rmse_results.loc[i, "training_error"] = np.sqrt(mean_squared_error(model.predict(polynomial_transform(X_train, degree)[features].values), y_train))
            rmse_results.loc[i, "test_error"] =np.sqrt(mean_squared_error(model.predict(polynomial_transform(X_val, degree)[features].values), y_val))
            i += 1
    return rmse_results