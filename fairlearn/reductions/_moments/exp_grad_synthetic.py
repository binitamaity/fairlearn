from fairlearn.reductions import BoundedGroupLoss, ExponentiatedGradient, ZeroOneLoss, SquareLoss
from noisy_bounded_group_loss import NoisyBoundedGroupLoss, ZeroOneLoss as ZL, SquareLoss,AbsoluteLoss
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from fairlearn.datasets import fetch_adult, fetch_boston
from scipy.stats import entropy

# , fetch_diabetes_hospital
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.linear_model import LinearRegression as LR
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))
# seed_value = 42
# random.seed(seed_value)

#set to 1 if noisy experiments are required
noisy_exp = 0



#For Synthetic Dataset
filename = 'new_synthetic_data_diff_mean_no_sa.csv'
X = pd.read_csv(filename)


print(X)
# assert 1 == 2


y_true = X['Label']

X = X.drop('Label', axis=1)

# print(y_true)
# print(X)
# assert 1 == 3



#Noise for Boston Dataset - taking LSTAT as the protected attribute

if noisy_exp:
    noisy_a = X['P_A_Noise']
else:
    noisy_a = np.ones(len(X['P_A_Noise']), dtype=int)

ent = entropy(noisy_a, base=2) / len(X['P_A_Noise'])
# print("Entropy : ",ent)
 
X = X.drop('P_A_Noise', axis=1)

pa_values = np.unique(X['P_A_most_likely'])
# print("pa vals : {}".format(pa_values))

#initializing the new column
X['protected_attribute']  = np.nan
X['P/Na'] = np.nan

#instantiate the points and add a column for storing P_ia/E[N_a] for each i
#If A == 'P_A_most_likely', then P_ia = 'P_A_Noise'
#If A != 'P_A_most_likely', then P_ia' = (1 - P_ia)(1/(num_groups-1))

def instantiate_pa(df, noisy_a):
    
    for i in range(len(df['protected_attribute'])):
        toss = np.random.choice([0,1], p =[noisy_a[i], 1-noisy_a[i]])

        if toss == 0:
            df.loc[i, 'protected_attribute'] = df.loc[i, 'P_A_most_likely']
            df.loc[i, 'P/Na'] = noisy_a[i]
        else:
            df.loc[i, 'protected_attribute'] = np.random.choice(np.setdiff1d(pa_values, df.loc[i, 'P_A_most_likely']))
            df.loc[i, 'P/Na'] = (1 - noisy_a[i])/(len(pa_values)-1)
    return df


def instantiate_pa_again(df, noisy_a):
    
    for i in range(len(df['protected_attribute.1'])):
        toss = np.random.choice([0,1], p =[noisy_a[i], 1-noisy_a[i]])

        if toss == 0:
            df.loc[i, 'protected_attribute'] = df.loc[i, 'P_A_most_likely']
            df.loc[i, 'P/Na'] = noisy_a[i]
        else:
            df.loc[i, 'protected_attribute'] = np.random.choice(np.setdiff1d(pa_values, df.loc[i, 'P_A_most_likely']))
            df.loc[i, 'P/Na'] = (1 - noisy_a[i])/(len(pa_values)-1)
    return df


X = instantiate_pa(X, noisy_a)
noisy_sa = X['protected_attribute']

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
# print(attr_counts)
# print(attr_counts[3])
# noisy_sa = X[['protected_attribute.1','protected_attribute.2', 'protected_attribute.3', 'protected_attribute.4',\
#               'protected_attribute.5','protected_attribute.6','protected_attribute.7','protected_attribute.8','protected_attribute.9',\
#                  'protected_attribute.10' ]]
print(X)
# noisy_sa = X[['protected_attribute.1', 'protected_attribute.2', 'protected_attribute.3']]
# print('Noisy_sa : ',noisy_sa)

per_group_losses = {}


# for i in range(len(pa_values)):
#     per_group_losses[i+1] = []

# bgl = BoundedGroupLoss(SquareLoss(0,1), upper_bound=1e-6)
bgl = NoisyBoundedGroupLoss(AbsoluteLoss(-1000,1000), upper_bound=2.0)


#Training

classifier1 = LR()
mitigator = ExponentiatedGradient(classifier1, bgl)
mitigator.fit(X.drop(['P_A_most_likely','P/Na','protected_attribute.1', 'protected_attribute.2', 'protected_attribute.3'], axis=1),\
               y_true, sensitive_features=noisy_sa, P_Na = X['P/Na']) #training with instantiated data


#Permuting the group labels to check correlation of true labels with group value
#This is the second instantiation, after training

# if not noisy_exp:
#     group_labels = X['protected_attribute']
#     # group_labels = np.random.permutation(group_labels)
#     group_labels = ((group_labels + 8)%10)+1
#     X['protected_attribute'] = group_labels
# noisy_sa = X['protected_attribute']
# print(X['protected_attribute'].value_counts())

# y_pred_mitigated = mitigator.predict(X)

# mae_noisy_mitigated = MetricFrame(metrics=mean_absolute_error, y_true=y_true, y_pred=y_pred_mitigated, sensitive_features=sa)
# mae_noisy_mitigated.overall

# mae_noisy_mitigated.by_group

if noisy_exp :
    for i in range(10):
        print(f"Iteration {i+1}")

        #2nd instantiation for evaluating
        X = instantiate_pa_again(X, noisy_a)
        noisy_sa = X[['protected_attribute']]
        # print(X['protected_attribute'].value_counts())

        y_pred_mitigated = mitigator.predict(X.drop(['P_A_most_likely', 'P/Na', 'protected_attribute.1', 'protected_attribute.2',\
                                                  'protected_attribute.3','protected_attribute'], axis=1))
        mae_noisy_mitigated = MetricFrame(metrics=mean_absolute_error, y_true=y_true, y_pred=y_pred_mitigated, sensitive_features=noisy_sa)
        print(mae_noisy_mitigated)
        print('Constraint - gamma:',bgl.gamma(lambda X: y_pred_mitigated)) 
        # print(mae_noisy_mitigated.overall)
        # print(mae_noisy_mitigated.by_group)
        losses = mae_noisy_mitigated.by_group
        losses = losses.values
        for l in range(len(pa_values)):
            per_group_losses[l+1].append(losses[l])

    # print(per_group_losses)

    instance = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    width = 0.08
    x = np.arange(len(instance))

    multiplier = 0

    fig, ax = plt.subplots()

    for attribute, measurement in per_group_losses.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        # ax.bar_label(rects, padding=10)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Loss')
    ax.set_title('Plot for Noisy Synthetic Dataset (_same_mean)')
    ax.set_xticks(x + 1 + width, instance)
    ax.set_xlabel('Number of Instantiation ', fontweight ='bold', fontsize = 15)
    ax.legend(loc='best', ncol=5)
    # ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()
    plt.savefig('Noisy_synthetic_same_mean.png')
else:
    y_pred_mitigated = mitigator.predict(X.drop(['P_A_most_likely', 'P/Na', 'protected_attribute.1', 'protected_attribute.2',\
                                                  'protected_attribute.3'], axis=1))
    mae_noisy_mitigated = MetricFrame(metrics=mean_absolute_error, y_true=y_true, y_pred=y_pred_mitigated, sensitive_features=noisy_sa)
    mae=[]
    mae_residual=[]
    y_true_np=y_true.to_numpy()
    # y_pred_np=y_pred_mitigated.to_numpy()
    for i in pa_values:
        diff = 0
        count=0
        res=0
        print('i',i)
        for j in range(len(X['P_A_most_likely'])):
            
            if X.loc[j,'P_A_most_likely']==i:
                count+=1
                diff+=np.abs(y_true_np[j]-y_pred_mitigated[j])
                res+=np.abs(X.loc[j,'Residual'])
                # print('diff:',np.abs(y_true_np[j]-y_pred_mitigated[j]))
        # print(count)
        mae_residual.append(np.sum(res)/count)
        mae.append(np.sum(diff)/count)
        
    print('residual_mae:',mae_residual)
    print('mae:',mae)
    print('bgl:',bgl.gamma(lambda X: y_pred_mitigated))
    losses=mae
    losses_1= mae_residual  
    # mae_noisy_mitigated = MetricFrame(metrics=mean_absolute_error, y_true=y_true, y_pred=y_pred_mitigated, sensitive_features=noisy_sa)
    # # print(mae_noisy_mitigated)
    # # print(y_pred_mitigated)
    # print('Constraint - gamma when upper bound is 0.01:',bgl.gamma(lambda X: y_pred_mitigated)) 
    # # print('bgl- bound',bgl.bound())
    # # print(mae_noisy_mitigated.overall)
    # # print(mae_noisy_mitigated.by_group)
    # losses = mae_noisy_mitigated.by_group
    # losses = losses.values

    # for l in range(len(pa_values)):
    #     per_group_losses[] = (losses[l], losses_1[l])
    per_group_losses['mae'] = losses
    per_group_losses['residual'] = losses_1

    # print(per_group_losses)


    # for l in range(len(pa_values)):
    #     per_group_losses[l+1].append(losses_1[l])

    # losses = bgl.gamma(lambda X: y_pred_mitigated)
    # print(type(bgl.gamma(lambda X: y_pred_mitigated)))
    # losses= losses.groupby
    # # losses = losses.values
    # for l in range(len(pa_values)):
    #     per_group_losses[l+1].append(losses[l])

    print(per_group_losses)

    instance = ['0','1','2']
    # instance=['mae','residual']
    width = 0.25
    x = np.arange(len(instance))

    multiplier = 0

    fig, ax = plt.subplots()

    # for attribute, measurement in per_group_losses.items():
    #     offset = width * multiplier
    #     # rects = ax.bar(x + offset, measurement, width, label=attribute)
        
    #     rects = ax.bar(x + offset, measurement, width, label=attribute)
    #     # rects = ax.bar(x + offset, measurement[1], width, label=attribute, alpha =0.5)
    #     ax.bar_label(rects, padding=3)
    #     multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Loss')
    # ax.set_title('MAE vs Residual per group  for fair regression')
    # ax.set_xticks(x+ width, instance)
    # ax.set_xlabel('Groups', fontweight ='bold', fontsize = 15)
    # ax.legend(loc='best', ncol=5)
    # ax.set_ylim(0, 5.5)

    # plt.tight_layout()
    # plt.show()
    # plt.savefig('non_noisy_synthetic_fair_regression_largeB_edited.png')
