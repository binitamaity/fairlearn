from fairlearn.reductions import ExponentiatedGradient, ZeroOneLoss, SquareLoss
# from noisy_bounded_group_loss import NoisyBoundedGroupLoss, ZeroOneLoss as ZL, SquareLoss
import random
from fairlearn.metrics import demographic_parity_ratio
from fairlearn.metrics import demographic_parity_difference
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from fairlearn.datasets import fetch_adult, fetch_boston
from scipy.stats import entropy

# , fetch_diabetes_hospital
from sklearn.metrics import mean_absolute_error
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


filename = 'new_synthetic_data_diff_mean_no_sa.csv'
X = pd.read_csv(filename)


print(X)
# assert 1 == 2
noisy_exp = 0

y_true = X['Label']

X = X.drop('Label', axis=1)

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



def one_hot_code(df1):
    cols = df1.columns
    for c in cols:
        if isinstance(df1[c][1], str) or isinstance(df1[c][1], int):
            column = df1[c]
            df1 = df1.drop(c, axis=1)
            unique_values = list(set(column))
            n = len(unique_values)
            if n > 2:
                for i in range(n):
                    col_name = '{}.{}'.format(c, i)
                    col_i = [1 if el == unique_values[i] else 0 for el in column]
                    df1[col_name] = col_i
            else:
                col_name = c
                col = [1 if el == unique_values[0] else 0 for el in column]
                df1[col_name] = col
    return df1

def log_numeric_features(df):
    cols = df.columns
    for c in cols:
        column =df[c]
        unique_values = list(set(column))
        n = len(unique_values)
        if n > 2:
            df[c] = np.log(1 + df[c])

    return df

X = one_hot_code(X)

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




X = instantiate_pa(X, noisy_a)
noisy_sa = X['protected_attribute']

per_group_losses = {}



print(X.shape[1])

classifier1 = LR()
mitigator = ExponentiatedGradient(classifier1, demographic_parity_difference)
mitigator.fit(X.drop(['P_A_most_likely'], axis=1),\
               y_true, sensitive_features=noisy_sa, P_Na = X['P/Na']) #training with instantiated data




y_pred_mitigated = mitigator.predict(X.drop(['P_A_most_likely', 'P_A_Noise','Residual','Label'], axis=1))
mae_noisy_mitigated = MetricFrame(metrics=mean_absolute_error, y_true=y_true, y_pred=y_pred_mitigated, sensitive_features=noisy_sa)



print(demographic_parity_difference(y_true,
                                    y_pred_mitigated,
                                    sensitive_features=noisy_sa))
print(demographic_parity_ratio(y_true, 
                               y_pred_mitigated,
                               sensitive_features=noisy_sa))


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
        
print('residual:',mae_residual)
print('mae:',mae)
print('SP:',demographic_parity_difference.gamma(lambda X: y_pred_mitigated))
losses=mae
losses_1= mae_residual  
per_group_losses['mae'] = losses
per_group_losses['residual'] = losses_1
print(per_group_losses)


instance = ['0','1','2']
# instance=['mae','residual']
width = 0.25
x = np.arange(len(instance))
multiplier = 0
fig, ax = plt.subplots()
for attribute, measurement in per_group_losses.items():
        offset = width * multiplier
        # rects = ax.bar(x + offset, measurement, width, label=attribute)
        
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        # rects = ax.bar(x + offset, measurement[1], width, label=attribute, alpha =0.5)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Loss')
ax.set_title('MAE vs Residual per group  for fair regression with SP')
ax.set_xticks(x+ width, instance)
ax.set_xlabel('Groups', fontweight ='bold', fontsize = 15)
ax.legend(loc='best', ncol=5)
# ax.set_ylim(0, 5.5)

plt.tight_layout()
plt.show()
plt.savefig('non_noisy_statistical_p_synthetic.png')
