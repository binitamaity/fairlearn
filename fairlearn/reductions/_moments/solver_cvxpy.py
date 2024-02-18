import numpy as np
import cvxpy as cp 
import pandas as pd
from fairlearn.reductions import BoundedGroupLoss, ExponentiatedGradient, ZeroOneLoss, SquareLoss
from noisy_bounded_group_loss import NoisyBoundedGroupLoss, ZeroOneLoss as ZL, SquareLoss,AbsoluteLoss
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt
from scipy.stats import entropy

import warnings
warnings.filterwarnings("ignore")

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

#For Synthetic Dataset
# filename = 'new_synthetic_data_diff_mean_no_sa.csv'
filename = 'new_synthetic_20_jan_2.csv'
X = pd.read_csv(filename)


y_true = X['Label']

# X = X.drop('Label', axis=1)


def one_hot_code(df1):
    cols = df1.columns
    for c in cols:
        # if isinstance(df1[c][1], str):
        if c == 'protected_attribute':
            column = df1[c]
            df1 = df1.drop(c,axis=1)
            unique_values = list(set(column))
            n = len(unique_values)
            if n > 2:
                for i in range(n):
                    col_name = '{}.{}'.format(c, i+1)
                    col_i = [1 if el == unique_values[i] else 0 for el in column]
                    df1[col_name] = col_i
            else:
                col_name = c
                col = [1 if el == unique_values[0] else 0 for el in column]
                df1[col_name] = col
    return df1

X  = one_hot_code(X)
print(X)

X1 = pd.DataFrame()
X2 = pd.DataFrame()
X3=  pd.DataFrame()


y1= pd.DataFrame()
y2= pd.DataFrame()
y3= pd.DataFrame()


pa_values = np.unique(X['P_A_most_likely'])
for i in pa_values:
    condition = (X['P_A_most_likely'] == i)
    if i==0:
        y1=X['Label'][condition].copy()
        # print('y1:',y1)
        X1=X[condition].copy()
        X1=X1.drop('Label', axis=1)
    if i==1:
        y2=X['Label'][condition].copy()
        X2=X[condition].copy()
        X2=X2.drop('Label', axis=1)
    if i==2:
        y3=X['Label'][condition].copy()
        X3=X[condition].copy()
        X3=X3.drop('Label', axis=1)  



X=X.drop(['P_A_most_likely', 'P_A_Noise','Residual','Label'], axis=1)
X=X.to_numpy()
# print('X:',X)
y_true_np=y_true.to_numpy()

n = X.shape[0]
d= X.shape[1]



X1=X1.drop(['P_A_most_likely', 'P_A_Noise','Residual'], axis=1)
X1=X1.to_numpy()
y1=y1.to_numpy()




X2=X2.drop(['P_A_most_likely', 'P_A_Noise','Residual'], axis=1)
X2=X2.to_numpy()
y2=y2.to_numpy()



X3=X3.drop(['P_A_most_likely', 'P_A_Noise','Residual'], axis=1)
X3=X3.to_numpy()
y3=y3.to_numpy()


# # Define and solve the CVXPY problem.
w_constraint = cp.Variable(d)
cost = (cp.sum_squares( X @ w_constraint  - y_true_np))/(X.shape[0])

cost1 = (cp.sum_squares( X1 @ w_constraint  - y1))/(X1.shape[0])
cost2 = (cp.sum_squares( X2 @ w_constraint  - y2))/(X2.shape[0])
cost3 = (cp.sum_squares( X3 @ w_constraint  - y3))/(X3.shape[0])

# w_constraint = cp.Variable(d)
# cost = (cp.sum_squares( X @ w_constraint  - y_true_np))

# cost1 = (cp.sum_squares( X1 @ w_constraint  - y1))
# cost2 = (cp.sum_squares( X2 @ w_constraint  - y2))
# cost3 = (cp.sum_squares( X3 @ w_constraint  - y3))


constraints = [cost1<=5, cost2<=5, cost3<=5]
# constraints = [cost2<=24]
 
prob = cp.Problem(cp.Minimize(cost), constraints)

prob.solve()


# print('prob:',prob.value)
# print((w_constraint.value))
# print('cost1:',cost1.value)
# print('cost:',mean_squared_error(X @ w_constraint.value,y_true_np))
print('cost1:',cost1.value)
print('cost2:',cost2.value)
print('cost3:',cost3.value)

#naive

w_naive = cp.Variable(d)
cost_n = (cp.sum_squares( X @ w_naive  - y_true_np))/(X.shape[0])

cost_1 = (cp.sum_squares( X1 @ w_naive  - y1))/(X1.shape[0])
cost_2 = (cp.sum_squares( X2 @ w_naive  - y2))/(X2.shape[0])
cost_3 = (cp.sum_squares( X3 @ w_naive  - y3))/(X3.shape[0])


 
prob1 = cp.Problem(cp.Minimize(cost_n))

prob1.solve()

print('cost_n:',cost_n.value)
print('cost_1:',cost_1.value)
print('cost_2:',cost_2.value)
print('cost_3:',cost_3.value)

# problem = cp.Problem(objective)

# # Solve the problem
# problem.solve()

