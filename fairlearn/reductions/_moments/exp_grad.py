from fairlearn.reductions import BoundedGroupLoss, ExponentiatedGradient, ZeroOneLoss, SquareLoss
from noisy_bounded_group_loss import NoisyBoundedGroupLoss, ZeroOneLoss as ZL, SquareLoss
import random
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



# ADULT DATASETS

# data = fetch_adult(as_frame=True,return_X_y= False)

#BOSTON DATASET
data = fetch_boston(as_frame=True, return_X_y=False)


X = data.data
for i in range(len(X['LSTAT'])):
    if (X.loc[i, 'LSTAT'] > (np.median(X['LSTAT'])/2)) and (X.loc[i, 'LSTAT'] <= np.median(X['LSTAT'])):
        X.loc[i, 'LSTAT'] = 1
    elif X.loc[i, 'LSTAT'] <= np.median(X['LSTAT'])/2:
        X.loc[i, 'LSTAT'] = 0
    else:
        X.loc[i, 'LSTAT'] = 2

X['LSTAT'] = X['LSTAT'].astype('int')
# X['LSTAT'] = X['LSTAT'].astype(str)

#For Synthetic Dataset
# filename = 'synthetic_data.csv'
# X = pd.read_csv(filename)

print(X)
# assert 1 == 2

#For Adult Dataset
# y_true = (data.target == ">50K") * 1

#For Boston Dataset
y_true = data.target

# print(y_true)
# assert 1 == 3

#Noise for Adult dataset
# noisy_a = np.random.random(len(X['sex']))
# noisy_a= np.ones(len(X['sex']))





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

# X = one_hot_code(X)
# X = log_numeric_features(X)
# print(X)

#1st instantiation
#For Adult Dataset
# X['sex'] = X['sex'].apply(lambda row: np.random.choice([0,1],p=[noisy_a[row],1-noisy_a[row]]))
# noisy_sa = X['sex']
pa_values = np.unique(X['LSTAT'])
print(pa_values)
#For Boston Dataset - instantiation
# X['LSTAT'] = X['LSTAT'].apply(lambda row: np.random.choice([0,1],p=[noisy_a[int(row)],1-noisy_a[int(row)]]))
# print(X['LSTAT'])
# assert 1 == 3
   
# bgl = BoundedGroupLoss(ZeroOneLoss(), upper_bound=0.01)
# bgl = BoundedGroupLoss(SquareLoss(0,1), upper_bound=1.0)
# 
noisy_sa = X['LSTAT']
print(noisy_sa)
# bgl.load_data(X, y_true, sensitive_features=noisy_sa)


#Training
# classifier1 = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)

classifier1 = LR()
# constraint = BoundedGroupLoss(ZeroOneLoss(), upper_bound=0.1)
constraint = BoundedGroupLoss(SquareLoss(-1,1), upper_bound=0.01)
mitigator = ExponentiatedGradient(classifier1, constraint)
mitigator.fit(X, y_true, sensitive_features=noisy_sa) #training with instantiated data
# constraint.load_data(X, y_true, sensitive_features=noisy_sa)


per_group_losses = {}


for i in range(len(pa_values)):
    per_group_losses[i+1] = []


# reg=LR().fit(X, y_true) 

y_pred_mitigated = mitigator.predict(X)
mae_noisy_mitigated = MetricFrame(metrics=mean_absolute_error, y_true=y_true, y_pred=y_pred_mitigated, sensitive_features=noisy_sa)
print(y_true[0])
print(y_pred_mitigated[0])
print('Constraint - gamma when upper bound is 0.01:',constraint.gamma(lambda X: y_pred_mitigated)) 
print('constraint- bound',constraint.bound())
print(mae_noisy_mitigated.overall)
losses = mae_noisy_mitigated.by_group
losses = losses.values
for l in range(len(pa_values)):
    per_group_losses[l+1].append(losses[l])

print(per_group_losses)

instance = ['1']
width = 1.0
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
ax.set_title('Regression plot for Boston dataset')
ax.set_xticks(x+1, instance)
ax.set_xlabel('Groups', fontweight ='bold', fontsize = 15)
ax.legend(loc='best', ncol=5)
# ax.set_ylim(0, 5.5)

plt.tight_layout()
plt.show()
plt.savefig('Boston_non_noisy_regression.png')






# datapoints = np.linspace(1,len(X['LSTAT']),len(X['LSTAT']))

# # Set position of bar on X axis
# fig, (ax0,ax1) = plt.subplots(2,1)
# ax0.bar(datapoints,noisy_a, color='r')
# ax0.set_xlabel('Datapoints')
# ax0.set_ylabel('Noise in measurement')

# br1 = np.arange(len(loss_1))
# br2 = [x + barWidth for x in br1]
# # br3 = [x + barWidth for x in br2]

# # Make the plot
# ax1.bar(br1, loss_1, color ='y', width = barWidth,
#     edgecolor ='grey', label ='LOW')
# ax1.bar(br2, loss_2, color ='b', width = barWidth,
#     edgecolor ='grey', label ='HIGH')


# # plt.bar(br3, CSE, color ='b', width = barWidth,
# # 		edgecolor ='grey', label ='CSE')

# # Adding Xticks
# fig.suptitle('Plot for Noisy Boston Dataset with entropy ' + str(ent))
# ax1.set_xlabel('Number of Instantiation', fontweight ='bold', fontsize = 15)
# ax1.set_ylabel('Loss', fontweight ='bold', fontsize = 15)
# ax1.set_xticks([r + barWidth for r in range(len(loss_1))],
#     ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

# ax1.legend()
# plt.show()
# # plt.savefig('Noisy_double_instance_adult10_'+str(ent)+'.png')
# plt.savefig('Noisy_double_instance_boston_lstat_1.png')




