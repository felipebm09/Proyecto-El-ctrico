# %% [markdown]
# ### Imports

# %%
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from pickle import dump
from sklearn.manifold import TSNE

# %% [markdown]
# ### Data Integration

# %%

print("Merging results for data integration...")
df1 = pd.read_excel("../Results/T_out1.xlsx")
df2 = pd.read_excel("../Results/T_out2.xlsx")
df3 = pd.read_excel("../Results/T_out3.xlsx")
df4 = pd.read_excel("../Results/T_out4.xlsx")
df5 = pd.read_excel("../Results/T_out5.xlsx")
df6 = pd.read_excel("../Results/T_out6.xlsx")
df7 = pd.read_excel("../Results/T_out7.xlsx")
df8 = pd.read_excel("../Results/T_out8.xlsx")
df9 = pd.read_excel("../Results/T_out9.xlsx")
df10 = pd.read_excel("../Results/T_out10.xlsx")
df11 = pd.read_excel("../Results/T_out11.xlsx")
df12 = pd.read_excel("../Results/T_out12.xlsx")
df13 = pd.read_excel("../Results/T_out13.xlsx")
df14 = pd.read_excel("../Results/T_out14.xlsx")

data = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14])

# data = pd.read_excel("../Results/ML_input.xlsx")
print('Done.')


# %% [markdown]
# ## Data Cleaning

# %%

print('Performing data cleaning...')
print('Inicial shape: ', data.shape)
print(data.columns)
# Rellenamos los espacios vacíos con cero
data = data.fillna(0)

# Elimino la columna de nombre del TIU
data = data.drop(['TIU'], axis=1)

# Eliminamos la fila de bin 1 ya es un bin de resultado positivo
data = data.drop(['1'], axis=1)

# Elimino las columnas que tengan socketing mejor a 30
index = data[data["Socketing"]<30].index
data = data.drop(index)
index = 0

# Elimino las colunas que no tengan algun valor mayor a 5
for col in data.columns:
    good_flag = 0
    for n in data.loc[:,col]:
        if(int(n) >= 5):
            good_flag = 1
    if((not good_flag) and (col != 'G/B_flag')):
        data = data.drop([col], axis=1)

print('Final Shape: ', data.shape)


# %%
# Características de los datos
data.describe().round(3)

# %% [markdown]
# ### Reducción de dimensionalidad
# Se busca correlation de los datos para reducir las dimensiones, recordando que para naive bayes lo mejor es que las características sean independientes entre si.

# %%
# calculate the correlations
correlations = data.corr()
correlations.to_excel("Correlation.xlsx")
# plot the heatmap 
sns.heatmap(correlations, xticklabels=correlations.columns, yticklabels=correlations.columns, annot=True)

# plot the clustermap 
sns.clustermap(correlations, xticklabels=correlations.columns, yticklabels=correlations.columns, annot=True)


# %% [markdown]
# Relaciones altas encontradas entre bines:
# 
# - bin 51 y bin13: 0.75
# - Bin 44 y bin 14: 0.73
# - Bin 44 y Bin 20: 0.71
# - Bin54 y Bin48: 0.88
# 
# Por experiencia del proceso sabemos que tanto el bin 13 como el 51 están asociados a fallos en este colateral por lo que decido dejar ambos como naive asumption
# se ve que el bin 44 representa bien la dinámica del bin 14 y el 20 por lo tanto se deja solo en 44
# y también por proceso sabemos que el bin54 está asociado a la TIU por lo tanto se eliminará el bin 48.

# %%
data = data.drop(['20'], axis=1)
data = data.drop(['14'], axis=1)
data = data.drop(['48'], axis=1)

print('Nueva dimension: ', data.shape)

# %%
data

# %% [markdown]
# ### Data Normalization

# %%
print('Begining with data normalization...')
Scaler = MinMaxScaler()
standard = StandardScaler()
buffer = data.copy()
columnas = data.columns
data = pd.DataFrame(Scaler.fit_transform(data), columns=columnas)
data_s = pd.DataFrame(standard.fit_transform(data), columns=columnas)

for i in range(len(data)):
    data.iloc[i, 1] = int(buffer.iloc[i, 1])
    data_s.iloc[i, 1] = int(buffer.iloc[i, 1])


# %%
# Guardamos scaler
dump(Scaler, open('scaler.pkl', 'wb'))

# %% [markdown]
# ### Visualizacion de datos

# %%
tsne = TSNE(n_components=2, random_state=0)
x_test_2d = tsne.fit_transform(data_s.drop(['G/B_flag'], axis=1))
labels = data_s['G/B_flag'].values
x_test_2d
plt.figure(figsize=(8,7))
plt.title("TIU marginal")
plt.scatter(x_test_2d[labels==0, 0], x_test_2d[labels == 0, 1], c='r', label='Bad', alpha=0.3)
plt.scatter(x_test_2d[labels==1, 0], x_test_2d[labels == 1, 1], c='b', label='Good', alpha=0.3)
plt.legend()
plt.show()

# %%
sns.pairplot(data.iloc[:,[0,1,2,4,5,9,10,23]], hue='G/B_flag')

# %% [markdown]
# Guardar resultados finales

# %%
# creating a new excel file and save the data
data.to_excel("../Results/ML_input.xlsx", index=False)


