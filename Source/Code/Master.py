# Archivo para aumatizar la lectura de bases de datos locales
import pandas as pd
import os
import time
# Medimos tiempo de ejecusión
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

fin = time.time()
print("Finished successfully!!")
print("Tiempo de ejecución: ", fin-inicio)