# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
=================================
Selection rates in census dataset
=================================
"""
from fairlearn.datasets import fetch_adult
import numpy as np
# %%
from fairlearn.metrics import MetricFrame, selection_rate
# from fairlearn.reductions import bounded_group_loss

# # SquareLoss, AbsoluteLoss, ZeroOneLoss 

data = fetch_adult(as_frame=True)
X = data.data
y_true = (data.target == ">50K") * 1
sex = X['sex']
# y_true = np.nan_to_num(y_true)
selection_rates = MetricFrame(
    metrics=selection_rate, y_true=y_true, y_pred=y_true, sensitive_features=sex
)

fig = selection_rates.by_group.plot.bar(
    legend=False, rot=0, title="Fraction earning over $50,000"
)
fig.figure.savefig("filename.png")
print(fig)
# import matplotlib.pyplot as plt

# plt.show()



