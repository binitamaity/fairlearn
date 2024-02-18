import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# Set the random seed for reproducibility
# np.random.seed(0)

# Define the number of rows and columns in your dataset

num_rows = [700,600,1200]
mean=[5,2,8]

protected_attr =[0,1,2]
num_columns = 10
num_protected_groups = 3
T=0
x=np.zeros((np.sum(num_rows),num_columns))
noisy_matrix=np.zeros((np.sum(num_rows),num_protected_groups))

print(x.shape)
# Create a random dataset , here x1 is x
for i in range(num_protected_groups):
    print(x[T:T+num_rows[i],:].shape)
    print(T)
    x[T:T+num_rows[i],:8] = np.random.normal(mean[i],1,size=(num_rows[i], num_columns-2))
    x[T:T+num_rows[i],8]=protected_attr[i]
    x[T:T+num_rows[i],9]=np.random.uniform(0.6,0.8,num_rows[i])
    T+=num_rows[i]
    
    
# print(x)



#idxs of the data
idxs = np.arange(0, np.sum(num_rows), 1)

# Create a DataFrame
df = pd.DataFrame(x, columns=['A', 'B','C','D','E','F','G','H','P_A_most_likely','P_A_Noise'])
print(df)

W = np.random.random_sample(df.shape[1]-2)
B=0
variance=[1,2,3]

df['Label'] = np.dot(df[['A', 'B','C','D','E','F','G','H']].values, W)
# print(df['Label'])


    
print(df.loc[0:10,'Label'])
for i in range(num_protected_groups):
    # print( "SHAPE",df.loc[B:B+num_rows[i],'Label'].shape)
    df.loc[B:B+num_rows[i]-1,'Residual'] = np.random.normal(0,variance[i],num_rows[i])
    df.loc[B:B+num_rows[i]-1,'Label'] += np.random.normal(0,variance[i],num_rows[i])
    
    B+=num_rows[i]

print(df)
df = shuffle(df)
print(df)
filename = 'new_synthetic_20_jan_2.csv'
df.to_csv(filename, index=False)