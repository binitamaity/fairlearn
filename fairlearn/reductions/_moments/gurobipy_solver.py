import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


#For Synthetic Dataset
filename = 'new_synthetic_data_diff_mean_no_sa.csv'
X = pd.read_csv(filename)


y_true = X['Label']
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
# print(X)

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
        print('y1:',y1)
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



# Create a Gurobi model

model = gp.Model("linear_regression")

# Add variables to the model
w_constraint = model.addMVar(X.shape[1], name="w_constraint")
print('w_cons:',w_constraint)

# error_vector = gp.quicksum((gp.tuplelist([(y_true_np[i] - X[i,:]@ w_constraint)*(y_true_np[i] - X[i,:]@ w_constraint) for i in range(len(y_true_np))])))





# # Set up the objective function (minimize the mean square error)
# model.setObjective(error_vector / X.shape[0], GRB.MINIMIZE)


# cost1 = (gp.quicksum((gp.tuplelist([(y1[i] - X1[i,:]@ w_constraint)*(y1[i] - X1[i,:]@ w_constraint) for i in range(len(y1))]))))
# cost2 = (gp.quicksum((gp.tuplelist([(y2[i] - X2[i,:]@ w_constraint)*(y2[i] - X2[i,:]@ w_constraint) for i in range(len(y2))]))))
# cost3 = (gp.quicksum((gp.tuplelist([(y3[i] - X3[i,:]@ w_constraint)*(y3[i] - X3[i,:]@ w_constraint) for i in range(len(y3))]))))


err = (gp.tuplelist(gp.abs_([(y_true_np[i] - X[i,:]@ w_constraint)]) for i in range(len(y_true_np))))
print("err",err)
error_vector = gp.quicksum(err)
# Set up the objective function (minimize the mean square error)
model.setObjective(error_vector / X.shape[0], GRB.MINIMIZE)


cost1 = (gp.quicksum(gp.abs_(gp.tuplelist([(y1[i] - X1[i,:]@ w_constraint) for i in range(len(y1))]))))
cost2 = (gp.quicksum(gp.abs_(gp.tuplelist([(y2[i] - X2[i,:]@ w_constraint) for i in range(len(y2))]))))
cost3 = (gp.quicksum(gp.abs_(gp.tuplelist([(y3[i] - X3[i,:]@ w_constraint) for i in range(len(y3))]))))




# # Add constraints constraints = [cost1<=0.1, cost2<=0.1, cost3<=0.1]
model.addConstr((cost1/ X1.shape[0])<=25, "constraint_1")
model.addConstr((cost2/ X2.shape[0])<=36, "constraint_2")
model.addConstr((cost3/ X3.shape[0])<=17, "constraint_3")

# Optimize the model
model.optimize()

# print(model.display())
# cost1=cost1/X1.shape[0]
print('cost1:',(cost1.getValue())/X1.shape[0])
print('cost2:',(cost2.getValue())/X2.shape[0])
print('cost3:',(cost3.getValue())/X3.shape[0])

# print('cost2:',cost2)

# print('cost3:',cost3)

