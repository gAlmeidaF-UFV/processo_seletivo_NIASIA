# -*- coding: utf-8 -*-
"""
Editor Spyder

Este é um arquivo de script temporário.
"""
#%% set_up
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import csv

features = ['first_column','second_column','third_column','fourth_column','fifth_column'
            ,'sixth_column','seventh_column','eith_column','nineth_column']

#%% lendo os arquivos e tranformando em valores
entradas_path = 'C:/Users/guilherme/Documents/NIASIA/entradasclassalunos.txt'
saida_path = 'C:/Users/guilherme/Documents/NIASIA/saidaclassalunos.txt'
entradas_csv_path ='C:/Users/guilherme/Documents/NIASIA/entradas.csv'
entradasteste_path = 'C:/Users/guilherme/Documents/NIASIA/entradasclassteste.txt'
entradasteste_path_csv = 'C:/Users/guilherme/Documents/NIASIA/entradasteste.csv'
previsoes = 'C:/Users/guilherme/Documents/NIASIA/GUILHERME_ALMEIDA_saidateste.txt'
model_csv_path = 'C:/Users/guilherme/Documents/NIASIA/model.csv'

def read_file(file_path):
    file_list = []
    with open(file_path,'r') as file_read:
        file_arq = file_read.readlines()
        for obj in file_arq: file_list.append(obj.split())
        if len(file_list[0]) == 1 : 
            for i in range(len(file_list)): file_list[i] = float(file_list[i][0])
    return file_list
            
def trans_to_csv(file_path_read, file_path_write):
    file_list = read_file(file_path_read)
    with open(file_path_write, 'w') as file_write:
        file_csv = csv.DictWriter(file_write, fieldnames = features)
        file_csv.writeheader()
        for i in range(len(file_list)):
            file_csv.writerow({features[j]:file_list[i][j] for j in range(len(file_list[0]))})

def write_txt(file_path_write, file_list):
    with open(file_path_write, 'w') as f:
        for i in range(len(file_list)):
            file_list[i] = round(file_list[i])
            f.write('\n')
            f.write('     %d'%file_list[i])        

saida_list = read_file(saida_path)
entradas_csv = trans_to_csv(entradas_path,entradas_csv_path)
    
#%% criando e treinando o modelo de IA
model_data = pd.read_csv(entradas_csv_path)
model_data['saida'] = saida_list
model_data = model_data.dropna(axis = 0)
model_data.to_csv(model_csv_path)

X = model_data[features]
y = model_data.saida

ia_model = RandomForestRegressor(random_state=1)
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=1)

ia_model.fit(train_X,train_y)
ia_model_preds = ia_model.predict(val_X)
for i in range(len(ia_model_preds)):
    ia_model_preds[i] = round(ia_model_preds[i])

mae_ia = mean_absolute_error(ia_model_preds,val_y)

print('O MAE do modelo IA é', mae_ia, '\n')

ia_model.fit(X, y)

#%% pruduzindo as predições

teste_csv = trans_to_csv(entradasteste_path, entradasteste_path_csv)

real_data = pd.read_csv(entradasteste_path_csv)
real_data = real_data.dropna(axis = 0)

ia_preds = ia_model.predict(real_data)

write_txt(previsoes, ia_preds)

print(ia_preds)
