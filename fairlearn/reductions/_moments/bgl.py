from fairlearn.reductions import BoundedGroupLoss, ZeroOneLoss
from fairlearn.datasets import fetch_adult
from sklearn.metrics import mean_absolute_error
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.linear_model import LinearRegression as LR
import numpy as np
data = fetch_adult(as_frame=False,return_X_y= False)
print("data: ",data)
X = data.data
# X.dropna()
y_true = (data.target == ">50K") * 1

# y_true = np.nan_to_num(y_true)
# selection_rates = MetricFrame(
#     metrics=selection_rate, y_true=y_true, y_pred=y_true, sensitive_features=sex
# )
X = np.nan_to_num(X)
sex = X[:,9]
print("SEX : ", sex)
# print("X", np.nan_to_num(X))
reg = LR().fit(X, y_true)
y_pred =reg.predict(X)

# fig = selection_rates.by_group.plot.bar(
#     legend=False, rot=0, title="Fraction earning over $50,000"
# )
bgl = BoundedGroupLoss(ZeroOneLoss(), upper_bound=0.1)

mae_frame = MetricFrame(metrics=mean_absolute_error,
                        y_true=y_true,
                        y_pred=y_pred,
                        sensitive_features=pd.Series(sex, name="SF 0"))
print("mae : ",mae_frame.overall)

mae_frame.overall
mae_frame.by_group
bgl.load_data(X, y_true, sensitive_features=sex)

print(bgl.gamma(lambda X: y_pred))