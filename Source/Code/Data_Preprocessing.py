import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

################################
####    DATA Integration    ####
################################

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



############################
####    DATA Cleaning   ####
############################
print('Performing data cleaning...')

# Rellenamos los espacios vacíos con cero
data = data.fillna(0)

# Eliminamos la fila de bin 1 ya es un bin de resultado positivo
data = data.drop(['1'], axis=1)

# Elimino las colunas que no tengan algun valor mayor a 5


# Elimino las columnas que tengan socketing mejor a 30

# Elimino la columna de nombre del TIU
data = data.drop(['TIU'], axis=1)


#################################
    DATA Normalization   ####
#####################################
print('Begining with data normalization...')
Scaler = MinMaxScaler()
col1 = data.iloc[:,0]
print(col1)
print(data)
data = Scaler.fit_transform(data)
print(data)
# data.iloc[:,0]=col1
# print(data)

# creating a new excel file and save the data
# data.to_excel("../Results/ML_input.xlsx", index=False)