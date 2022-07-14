# %% [markdown]
# ## Proyecto Eléctrico - HDMX early bird
# ### Autor: Felipe Badilla Marchena - B70848
# ## agvs: 1: Archivo a leer, 1: Archivo donde escribir resultado, 3: Relación de proporción

# %% [markdown]
# #### Preprocesamiento de datos:

# %%
# Librerías a utilizar

import pandas as pd
import numpy as np
import os
import sys
print('Running preprocessing...')

# %%
# Manejo de archivos
file_name = str(sys.argv[1])

excel_filename = file_name
print('Analyzing document: ',excel_filename)

# Se importan los datos
data = pd.read_excel(excel_filename)

df = pd.DataFrame(data, columns= ['VISUAL_ID', 'WITHIN_SESSION_SEQUENCE_NUMBER',
                                'WITHIN_SESSION_LATEST_FLAG', 'INTERFACE_BIN',
                                'TESTER_INTERFACE_UNIT_ID', 'THERMAL_HEAD_ID',
                                'DEVICE_TESTER_ID', 'MODULE', 'SITE_ID', 'TEST_TIME',
                                'DEVICE_END_DATE_TIME', 'LOT'])
# Ordenamos por fecha y hora
df = df.sort_values(by=['DEVICE_END_DATE_TIME'])
df = df.reset_index(drop=True)
df.shape


# %% [markdown]
# #### Filtrado de bines sólidos y conteo de bines por colateral general y por lote

# %%
# Recorremos el array para ver sólidos y los eliminaremos de la lista (lista nueva llamada df_Switching)

# Solid_index = []
Solid_visual = []
SV1 = []
SV3 = []
SV5 = []

Current_retest_index = []

df = df.sort_values(by=['VISUAL_ID','WITHIN_SESSION_SEQUENCE_NUMBER'])          # Ordenamos por visual y número de secuencia
df_Switching = df.copy()

diferent_flag = 0
bin_switch_flag = 0

prev_visual = ''
prev_bin = 0
current_bin = 0
current_visual = ''

total_units = 0

print('Buscando unidades sólidas!')
# Recorremos los datos
for index, row in df.iterrows():
    prev_bin = current_bin
    prev_visual = current_visual

    current_visual = row["VISUAL_ID"]
    current_bin = row["INTERFACE_BIN"]
    
    if prev_visual != current_visual:
        diferent_flag = 1
    else:
        diferent_flag = 0
    
    # Nueva unidad
    if diferent_flag == 1:
        prev_bin = 0
        total_units = total_units + 1
        bin_switch_flag = 0
        
        if(current_bin == 1):
            # Unidades buenas, no las tomamos en cuenta
            # Solid_index.append(index)
            if(row['MODULE'] == 'HXV101'):
                SV1.append(current_visual)
            elif(row['MODULE'] == 'HXV103'):
                SV3.append(current_visual)
            elif(row['MODULE'] == 'HXV105'):
                SV5.append(current_visual)
        else:
            # Unidad de retest, hay que analizar
            Current_retest_index.append(index)

    # Unidad de retest
    if diferent_flag == 0:
        Current_retest_index.append(index)

        # Es fallo sólido
        if (prev_bin == current_bin) and (bin_switch_flag == 0) and (row['WITHIN_SESSION_LATEST_FLAG'] == 'Y'):
            # Solid_index = Solid_index + Current_retest_index            # Sumamos a los sólidos 
            Current_retest_index.clear()                                # Vaciamos buffer de index
            if(row['MODULE'] == 'HXV101'):
                SV1.append(current_visual)
            elif(row['MODULE'] == 'HXV103'):
                SV3.append(current_visual)
            elif(row['MODULE'] == 'HXV105'):
                SV5.append(current_visual)

        # Es bin Switch
        if (prev_bin != current_bin) or (bin_switch_flag == 1):
            bin_switch_flag = 1                                     # Prendemos bandera para que no se confunda con solidas
            if (row["WITHIN_SESSION_LATEST_FLAG"] == 'Y') :
                Current_retest_index.clear()                        # Vaciamos el buffer de index
Solid_visual = [SV1, SV3, SV5]
print('Se van a ignorar: ', int(len(Solid_visual[0])+int(len(Solid_visual[1]))+int(len(Solid_visual[2]))), ' líneas que son unidades sólidas.')
df_Switching = df
df_Switching.shape


# %% [markdown]
# ## Filtrado:
# 
# - Conteo en una sola celda por un solo colateral
# - Posteriormente a las demás celdas
# - Repote para los demás tools

# %% [markdown]
# ##### Contamos "socketing" por colateral y promedio de test time

# %%

df_Switching_Backup = df_Switching.copy()

# Dataframe de salida del preprocesamiento de datos y entrada del algoritmo de ML
df_final = pd.DataFrame(columns=['Socketing', 'TIU', 'G/B_flag',
    'Test_Time', 'Bines_General', 'Bines_NLot',
    '1' , '2', '3', '4', '5', '6', '7', '8', '9', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
    '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
    '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
    '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
    '61', '62', '63', '64', '65', '66', '67', '68', '69', '70',
    '71', '72', '73', '74', '75', '76', '77', '78', '79', '80',
    '81', '82', '83', '84', '85', '86', '87', '88', '89', '90',
    '91', '92', '93', '94', '95', '96', '97', '98', '99'])  

# Listas de maquinas y celdas donde se busca la información de performance por colateral.
Tool_Number = ['HXV101', 'HXV103', 'HXV105']

Cell_Number = ['A101', 'A102', 'A201', 'A202', 'A301', 'A302', 'A401', 'A402', 'A501', 'A502',
            'B101', 'B102', 'B201', 'B202', 'B301', 'B302', 'B401', 'B402', 'B501', 'B502', 
            'C101', 'C102', 'C201', 'C202', 'C301', 'C302', 'C401', 'C402', 'C501', 'C502']

n = 0       # Indice de fila a rellenar
prev_TIU = ''
current_TIU = ''
counter =0

# Recorremos el array para cada tool
for Tool in Tool_Number:

    print('Analizing tool ', Tool, '...')
    df_tool = df[df.MODULE.isin([Tool])]
    df_Switching_tool = df_Switching_Backup[df_Switching_Backup.MODULE.isin([Tool])]

    # Iteración por celda
    for Cell in Cell_Number:  
        print('Analizing cell ', Cell, '...')
        df_celda = df_tool[df_tool.SITE_ID.isin([Cell])]
        df_Switching = df_Switching_tool[df_Switching_tool.SITE_ID.isin([Cell])]

        # Primero lo haré para TIU, Aquí cambiaríamos a otros colaterales
        df_TIU = df_celda.drop(['THERMAL_HEAD_ID'], axis=1)
        df_TIU = df_TIU.drop(['DEVICE_TESTER_ID'], axis=1)
        df_TIU = df_TIU.drop(['MODULE'], axis=1)
        df_TIU = df_TIU.drop(['SITE_ID'], axis=1)
        
        df_TIU = df_TIU.sort_index()
        df_Switching = df_Switching.sort_index()
        current_TIU_socketing = 1
        test_time_prom = 0
        DUT_index = []          #Aqui guardo los indices de la TIU que estamos analizando para luego sacar muestras buenas.
        solid_search = 0        # To change search of solid visual between tools

        #################################
                                        #
        balance = int(sys.argv[3])      # Este valor quiere decir cuantos datos "Buenos" habrán por cada "Malo" EJ: 10/1
        contador_balance = 0            # variable auxiliar de balance
        relacion_balance = 0            # variable auxiliar de balance
        relacion_balance_dinamico = 0   # variable auxiliar de balance
                                        #
        #################################
        

        # Recorremos el array y detectamos cambios de colateral (que suponemos como marginalidad)
        for index, row in df_TIU.iterrows():
            prev_LOT = ''
            current_LOT = ''
            Lot_History = [0]
            prev_TIU = current_TIU
            current_TIU = row['TESTER_INTERFACE_UNIT_ID']
            current_Test_time= row['TEST_TIME']
            current_date = row['DEVICE_END_DATE_TIME']

            test_time_prom = test_time_prom + current_Test_time

            # Seguimos en el mismo colateral
            if prev_TIU == current_TIU:
                DUT_index.append(index)
                current_TIU_socketing = current_TIU_socketing + 1

            # Diferente colateral, nueva fila
            elif (prev_TIU != current_TIU) and (prev_TIU != ''):
                df_final.loc[str(n)] = np.zeros(105)
                df_final.loc[str(n), 'Socketing'] = current_TIU_socketing
                df_final.loc[str(n), 'TIU'] = prev_TIU[-5:]
                df_final.loc[str(n), 'G/B_flag'] = 0
                df_final.loc[str(n), 'Test_Time'] = round(test_time_prom / current_TIU_socketing, 3)
                
                # Buscamos bines (ignoramos sólidos)
                reg_index = []
                a_flag = 1

                relacion_balance = current_TIU_socketing/(balance + 2)  # Le sumamos 1 al balance para evitar el caso donde ya está malo el colateral
                contador_balance = 0            # variable auxiliar de balance
                relacion_balance_dinamico = relacion_balance

                for index_2, row_2 in df_Switching.iterrows():
                    counter += 1
                    prev_LOT = current_LOT
                    current_LOT = row_2['LOT']

                    #####       Parte del código para calcular filas de colaterales buenos      #####
                    
                    if((contador_balance == int(relacion_balance_dinamico)) and (int(relacion_balance_dinamico) <= int(relacion_balance*(balance)))):
                        df_final.loc[str(n), 'G/B_flag'] = 1                                    # Se indica que es unidad buena
                        df_final.loc[str(n), 'Socketing'] = int(relacion_balance_dinamico)
                        df_final.loc[str(n), 'Test_Time'] = row_2['TEST_TIME']                  # usamos test time actual (todos derían ser similares)
                        df_final.loc[str(n), 'Bines_NLot'] = round(sum(Lot_History)/len(Lot_History), 3)             # Metemos el promedio de bines de fallo por lote
                        n += 1
                        df_final.loc[str(n)] = np.zeros(105)                                    # creamos nueva fila
                        df_final.loc[str(n)] = df_final.loc[str(n-1)]                           # copiamos la anterior
                        df_final.loc[str(n), 'Socketing'] = current_TIU_socketing               # Rellenamos valores por defecto (el malo)
                        df_final.loc[str(n), 'G/B_flag'] = 0
                        df_final.loc[str(n), 'Test_Time'] = round(test_time_prom / current_TIU_socketing, 3)
                        relacion_balance_dinamico = relacion_balance_dinamico + relacion_balance
                    else:
                        contador_balance = contador_balance + 1

                    #####                                                                       #####

                    if (row_2['TESTER_INTERFACE_UNIT_ID'] == prev_TIU) and (a_flag == 1):
                        reg_index.append(index_2)

                        if(str(row_2['VISUAL_ID']) not in Solid_visual[solid_search]):
                            
                            df_final.loc[str(n), str(int(row_2['INTERFACE_BIN']))] += 1         # sumamos a los bines 
                            df_final.loc[str(n), 'Bines_General'] += 1                          # Sumamos al historial general de bines

                            # Verificamos por lote (hacer promedio por lote)
                            if(current_LOT == prev_LOT):
                                Lot_History[-1] += 1                # Sumamos a los bines de un mismo lote
                            else:
                                Lot_History.append(1)               # Agregamos un lote nuevo
                        else:
                            Solid_visual[solid_search].remove(str(row_2['VISUAL_ID']))                        # Eliminamos el visual que ya fue consultado
                    else:
                        a_flag = 0
                df_Switching_Backup = df_Switching_Backup.drop(reg_index)
                df_Switching = df_Switching.drop(reg_index)
                
                df_final.loc[str(n), 'Bines_NLot'] = round(sum(Lot_History)/len(Lot_History), 3)             # Metemos el promedio de bines de fallo por lote
                
                test_time_prom = 0
                n = n+1
                current_TIU_socketing = 1
                reg_index = []

        # Temporalmente terminamos con un cierto colateral que suponemos como bueno
        df_final.loc[str(n)] = np.zeros(105)
        df_final.loc[str(n),'Socketing'] = current_TIU_socketing
        df_final.loc[str(n), 'TIU'] = current_TIU[-5:]
        df_final.loc[str(n), 'G/B_flag'] = 1
        df_final.loc[str(n), 'Test_Time'] = round(test_time_prom / current_TIU_socketing, 3)
        a_flag = 1
        prev_LOT = ''
        current_LOT = ''
        Lot_History = [0]
        reg_index = []

        # Buscamos bines 
        for index_2, row_2 in df_Switching.iterrows():
            counter += 1
            prev_LOT = current_LOT
            current_LOT = row_2['LOT']
            
            if (row_2['TESTER_INTERFACE_UNIT_ID'] == prev_TIU) and (a_flag == 1):
                reg_index.append(index_2)

                if(str(row_2['VISUAL_ID']) not in Solid_visual[solid_search]):

                    df_final.loc[str(n), str(int(row_2['INTERFACE_BIN']))] += 1             # sumamos a los bines 
                    df_final.loc[str(n), 'Bines_General'] += 1                              # Sumamos al historial general de bines
                    
                    # Verificamos por lote (hacer promedio por lote)
                    if(current_LOT == prev_LOT):
                        Lot_History[-1] += 1                # Sumamos a los bines de un mismo lote
                    else:
                        Lot_History.append(1)               # Agregamos un lote nuevo
                else:
                    Solid_visual[solid_search].remove(str(row_2['VISUAL_ID']))                        # Eliminamos el visual que ya fue consultado
            else:
                a_flag = 0
        df_final.loc[str(n), 'Bines_NLot'] = round(sum(Lot_History)/len(Lot_History), 3)             # Metemos el promedio de bines de fallo por lote
        test_time_prom = 0
        df_Switching_Backup = df_Switching_Backup.drop(reg_index)
        df_Switching = df_Switching.drop(reg_index)

        n = n+1
        current_TIU_socketing = 1
    # print(len(df_final))
    solid_search = solid_search + 1
    print("Done!")
    
# print(counter)
total_analized = df_final['Socketing'].sum()
df_final


# %% [markdown]
# ### Filtramos condiciones especiales de los datos finales

# %%
# Elimino todas las columnas que son cero (bines que nunca ocurrieron)
for o in range(99):
    if(df_final[str(o+1)].sum() == 0):
        del df_final[str(o+1)]

# Elimino filas con colaterales nulos (no se corrieron unidades en la celda)
# Elimino filas con cero bines malos, estos indican que todos los malos que tuvieron fueron sólidos
# Eliminio filas "Buenas" que son los ultimos colaterales en módulo (no se pueden clasificar aun)
# Elimino las filas con socketing cero o con socketing repetido (para las muestras de colaterales buenos)
empy_row = []
prev_socketing = 0
for index, row in df_final.iterrows():
    # print(row['Socketing'])
    # print(prev_socketing)
    if row['TIU'] == '':
        empy_row.append(index)
    elif row['Bines_General'] == 0:
        empy_row.append(index)
    elif (((row['G/B_flag'] == 1) and (row['Socketing'] == prev_socketing)) | (row['Socketing'] == 0)):
        empy_row.append(index)
    prev_socketing = row['Socketing']
df_final = df_final.drop(empy_row)

df_final

# %% [markdown]
# Escribimos el archivo con los datos listos para alimentar el modelo

# %%
# Archivo en donde escribimos los resultados
# file_results_name = '\ML_input.xlsx'
excel_results_file = sys.argv[2]

# crear el objeto ExcelWriter
escrito = pd.ExcelWriter(excel_results_file)

# escribir el DataFrame en excel
print('Writing results file...')
df_final.to_excel(escrito, index=False)

# guardar el excel
escrito.save()
print('DONE')
