from fairlearn.reductions import BoundedGroupLoss, ZeroOneLoss as ZL
from noisy_bounded_group_loss import NoisyBoundedGroupLoss, ZeroOneLoss
from fairlearn.datasets import fetch_adult
from sklearn.metrics import mean_absolute_error
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.linear_model import LinearRegression as LR
import numpy as np
data = fetch_adult(as_frame=True,return_X_y= False)
# print("data: ",data)
X = data.data
# X.dropna()
y_true = (data.target == ">50K") * 1
# print("TRUE", y_true)
noisy_a= np.random.uniform(low=0.4, high=0.6, size=len(X['sex']))
# noisy_a = np.random.rand(len(X['sex']))
print("Noise", noisy_a)




# y_true = np.nan_to_num(y_true)
# selection_rates = MetricFrame(
#     metrics=selection_rate, y_true=y_true, y_pred=y_true, sensitive_features=sex
# )
# X = np.nan_to_num(X)
# X['sex'] = X['sex'].apply(lambda row: np.random.choice(1,p=[noisy_a[row],1-noisy_a[row]]))

# X['sex'] = np.random.choice(1, p = noisy_a) 
# sex = X['sex']
# print("SEX : ", sex)


# print("X", np.nan_to_num(X))
# reg = LR().fit(X, y_true)
# y_pred =reg.predict(X)


def one_hot_code(df1):
    cols = df1.columns
    for c in cols:
        if isinstance(df1[c][1], str):
            column = df1[c]
            df1 = df1.drop(c, 1)
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

X = one_hot_code(X)
log_numeric_features(X)



# fig = selection_rates.by_group.plot.bar(
#     legend=False, rot=0, title="Fraction earning over $50,000"
# )

bgl = BoundedGroupLoss(ZL(), upper_bound=0.1)

# mae_frame = MetricFrame(metrics=mean_absolute_error,
#                         y_true=y_true,
#                         y_pred=y_true,
#                         sensitive_features=pd.Series(sex, name="SF 0"))
# print("mae : ",mae_frame.overall)

# mae_frame.overall
# mae_frame.by_group

sa = X['sex']
bgl.load_data(X, y_true, sensitive_features=sa)

reg = LR().fit(X, y_true)
# y_pred =reg.predict(X)

# print(bgl.gamma(lambda X: y_true))
print('Definite : ', bgl.gamma(lambda X: reg.predict(X)))


X['sex'] = X['sex'].apply(lambda row: np.random.choice([0,1],p=[noisy_a[row],1-noisy_a[row]]))
noisy_sa = X['sex']
print("SEX : ", noisy_sa)

noisy_reg = LR().fit(X, y_true)
nbgl = NoisyBoundedGroupLoss(ZeroOneLoss(), upper_bound=0.1)
nbgl.load_data(X, y_true, sensitive_features=noisy_sa)
print('Noisy : ',nbgl.gamma(lambda X: noisy_reg.predict(X)))