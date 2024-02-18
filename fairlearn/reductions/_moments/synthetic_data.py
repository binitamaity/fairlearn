import pandas as pd
import numpy as np

# Set the random seed for reproducibility
# np.random.seed(0)

# Define the number of rows and columns in your dataset
num_rows = 5000
num_columns = 8
num_protected_groups = 10

# Create a random dataset
data = np.random.random(size=(num_rows, num_columns))

#idxs of the data
idxs = np.arange(0, num_rows, 1)

# 1% of the indices
idx_80pc = np.random.randint(0, num_rows, int(0.8*num_rows))
print(idx_80pc)
rem_idxs = np.setdiff1d(idxs, idx_80pc) 
print(rem_idxs)

ones_vector = np.ones(idx_80pc.shape[0], dtype=int)

idx_2pc = np.random.choice(rem_idxs, int(0.02*len(rem_idxs)))
tens_vector = 10 * np.ones(idx_2pc.shape[0], dtype=int)

rem_idxs = np.setdiff1d(rem_idxs, idx_2pc)


# Create a DataFrame
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E', 'F','G', 'H'])
df['P_A_most_likely'] = np.nan
df.loc[idx_80pc,'P_A_most_likely'] = ones_vector
df.loc[idx_2pc, 'P_A_most_likely'] = tens_vector

for i in rem_idxs:
    df.loc[i, 'P_A_most_likely'] = int(np.random.randint(2,high=10,size=1))


W = np.random.random_sample(df.shape[1])

df['Label'] = np.dot(df.values, W)

for i in range(len(df['Label'])):
    df.loc[i, 'Label'] += np.random.randn()

# make the labels nonlinear
# df['Label'] = np.abs(np.sin(df['Label']))


#This column includes the probability of that row
#having the most likely value of the protected attribute
df['P_A_noise'] = np.nan
P_A_noise = np.random.uniform(0.4,0.6,len(df['P_A_noise']))
df.loc[idxs, 'P_A_noise'] = P_A_noise


# Display the DataFrame
print(df)
print(df['P_A_most_likely'].value_counts())

filename = 'synthetic_data.csv'
df.to_csv(filename, index=False)
