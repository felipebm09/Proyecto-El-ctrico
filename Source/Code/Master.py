import pandas as pd
import sys
import os
import time
inicio = time.time()

DATA_DIR = "\Raw_Data"
RESULTS_DIR = "\Results"

Inputs_DB = ["\DB1.xlsx", "\DB2.xlsx", "\DB3.xlsx", "\DB4.xlsx", "\DB5.xlsx", "\DB6.xlsx", "\DB7.xlsx", "\DB8.xlsx", "\DB9.xlsx", "\DB10.xlsx", "\DB11.xlsx", "\DB12.xlsx", "\DB13.xlsx", "\DB14.xlsx"]
Output_DB = ["\T_out1.xlsx", "\T_out2.xlsx", "\T_out3.xlsx", "\T_out4.xlsx", "\T_out5.xlsx", "\T_out6.xlsx", "\T_out7.xlsx", "\T_out8.xlsx", "\T_out9.xlsx", "\T_out10.xlsx", "\T_out11.xlsx", "\T_out12.xlsx", "\T_out13.xlsx", "\T_out14.xlsx"]
Preprocessing_file = "C:/Users/felip/AppData/Local/Programs/Python/Python39/python.exe Data_Transformation.py"
current_dir = os.getcwd()
Rutabase = os.path.abspath(os.path.join(current_dir, os.pardir))
contador = 0
for inputDB in Inputs_DB:
    print("Analizing DataBase...")
    Rutarel_in = DATA_DIR + inputDB
    input = Rutabase + Rutarel_in
    Rutarel_out = RESULTS_DIR +  Output_DB[contador]
    contador += 1
    output = Rutabase + Rutarel_out
    balance = "5"
    print('Input: ', input)
    os.system(Preprocessing_file + ' ' + input + ' ' + output + ' ' + balance)

print("Merging results")
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

df_final = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14])

# creating a new excel file and save the data
df_final.to_excel("../Results/ML_input.xlsx", index=False)

# os.remove("../Results/T_out1.xlsx")
# os.remove("../Results/T_out2.xlsx")
# os.remove("../Results/T_out3.xlsx")
# os.remove("../Results/T_out4.xlsx")
# os.remove("../Results/T_out5.xlsx")
# os.remove("../Results/T_out6.xlsx")
# os.remove("../Results/T_out7.xlsx")
# os.remove("../Results/T_out8.xlsx")
# os.remove("../Results/T_out9.xlsx")
# os.remove("../Results/T_out10.xlsx")
# os.remove("../Results/T_out11.xlsx")
# os.remove("../Results/T_out12.xlsx")
# os.remove("../Results/T_out13.xlsx")
# os.remove("../Results/T_out14.xlsx")
fin = time.time()
print("Finished successfully!!")
print("Tiempo de ejecución: ", fin-inicio)