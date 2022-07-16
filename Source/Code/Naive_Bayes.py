# %% [markdown]
# # Naive Bayes Classifier

# %%
# import libraries
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.naive_bayes import MultinomialNB                                   # Para Modelo de Naive Bayers Multinomial
from sklearn.naive_bayes import GaussianNB                                      # Para Modelo de Naive Bayers Multinomial
from sklearn.naive_bayes import BernoulliNB                                     # Para Modelo de Naive Bayers Multinomial
from sklearn.metrics import recall_score, precision_score, f1_score             # Para medir rendimiento del modelo
from pickle import dump
from sklearn.model_selection import KFold

# %%
# importar datos
# Manejo de archivos
file_name = 'ML_input.xlsx'
current_dir = os.getcwd()
Rutabase = os.path.abspath(os.path.join(current_dir, os.pardir))
RESULTS_DIR = "\Results"
excel_filename = Rutabase + RESULTS_DIR +  '\ML_input.xlsx'
print('Analyzing document: ',excel_filename)
data = pd.read_excel(excel_filename)
data

# %%
# Separar datos entre entrenamiento y pruebas
tag = data['G/B_flag']
data_train = data.drop(['G/B_flag'], axis=1)
print(data_train.shape)
print(tag.shape)

# %%
# Creación de modelo
classifier_multinomial = MultinomialNB()
classifier_gaussian = GaussianNB()
classifier_bernuilli = BernoulliNB()

# Entrenamiento del modelo
classifier_multinomial.fit(data_train, tag)
classifier_gaussian.fit(data_train, tag)
classifier_bernuilli.fit(data_train, tag)

# save the model
dump(classifier_multinomial, open('model_MULT.pkl', 'wb'))
dump(classifier_bernuilli, open('model_BERNU.pkl', 'wb'))
dump(classifier_gaussian, open('model_GAUS.pkl', 'wb'))

# %% [markdown]
# ### Cross Validation

# %%
kf = KFold(n_splits = 10)
Bernuilli_score = []
Bernuilli_recall =[]
Bernuilli_precision = []
Bernuilli_f1 = []

Gaussian_score = []
Gaussian_recall =[]
Gaussian_precision = []
Gaussian_f1 = []

Multinomial_score = []
Multinomial_recall =[]
Multinomial_precision = []
Multinomial_f1 = []

for train_index, test_index in kf.split(data):
    # Recalculando modelos
    classifier_multinomial.fit(data_train.iloc[train_index], data.iloc[train_index,1])
    classifier_bernuilli.fit(data_train.iloc[train_index], data.iloc[train_index,1])
    classifier_multinomial.fit(data_train.iloc[train_index], data.iloc[train_index,1])

    # Rendimiento multinomial
    Multinomial_recall.append(recall_score(tag.iloc[test_index], classifier_multinomial.predict(data_train.iloc[test_index])))
    Multinomial_score.append(classifier_multinomial.score(data_train.iloc[test_index], tag.iloc[test_index]))
    Multinomial_precision.append(precision_score(tag.iloc[test_index], classifier_multinomial.predict(data_train.iloc[test_index])))
    Multinomial_f1.append(f1_score(tag.iloc[test_index], classifier_multinomial.predict(data_train.iloc[test_index])))

    # Rendimiento Gaussian
    Gaussian_recall.append(recall_score(tag.iloc[test_index], classifier_gaussian.predict(data_train.iloc[test_index])))
    Gaussian_score.append(classifier_gaussian.score(data_train.iloc[test_index], tag.iloc[test_index]))
    Gaussian_precision.append(precision_score(tag.iloc[test_index], classifier_gaussian.predict(data_train.iloc[test_index])))
    Gaussian_f1.append(f1_score(tag.iloc[test_index], classifier_gaussian.predict(data_train.iloc[test_index])))

    # Rendimiento Bernuilli
    Bernuilli_recall.append(recall_score(tag.iloc[test_index], classifier_bernuilli.predict(data_train.iloc[test_index])))
    Bernuilli_score.append(classifier_bernuilli.score(data_train.iloc[test_index], tag.iloc[test_index]))
    Bernuilli_precision.append(precision_score(tag.iloc[test_index], classifier_bernuilli.predict(data_train.iloc[test_index])))
    Bernuilli_f1.append(f1_score(tag.iloc[test_index], classifier_bernuilli.predict(data_train.iloc[test_index])))

print('Multinomial score: ', round(sum(Multinomial_score) / len(Multinomial_score),3)*100)
print('Multinomial recall: ', round(sum(Multinomial_recall) / len(Multinomial_recall),3)*100)
print('Multinomial presicion: ', round(sum(Multinomial_precision) / len(Multinomial_precision),3)*100)
print('Multinomial f1: ', round(sum(Multinomial_f1) / len(Multinomial_f1),2)*100)

print('----------------------------------')

print('Gaussian score: ', round(sum(Gaussian_score) / len(Gaussian_score),3)*100)
print('Gaussian recall: ', round(sum(Gaussian_recall) / len(Gaussian_recall),3)*100)
print('Gaussian presicion: ', round(sum(Gaussian_precision) / len(Gaussian_precision),3)*100)
print('Gaussian f1: ', round(sum(Gaussian_f1) / len(Gaussian_f1),3)*100)

print('----------------------------------')

print('Bernuilli score: ', round(sum(Bernuilli_score) / len(Bernuilli_score),3)*100)
print('Bernuilli recall: ', round(sum(Bernuilli_recall) / len(Bernuilli_recall),3)*100)
print('Bernuilli presicion: ', round(sum(Bernuilli_precision) / len(Bernuilli_precision),3)*100)
print('Bernuilli f1: ', round(sum(Bernuilli_f1) / len(Bernuilli_f1),3)*100)

# %% [markdown]
# ### Estadísticas

# %%
## Declaramos valores para el eje x
eje_x = ['Multinomial', 'Gaussian', 'Bernuilli']

eje_y = [round(sum(Multinomial_score) / len(Multinomial_score),3)*100, round(sum(Gaussian_score) / len(Gaussian_score),3)*100, round(sum(Bernuilli_score) / len(Bernuilli_score),3)*100]
plt.bar(eje_x, eje_y, color=['blue','red','green'])
plt.ylabel('Score')
plt.xlabel('Modelo')
plt.title('Naive Bayes')
plt.show()

# %%
 
eje_y = [ round(sum(Multinomial_recall) / len(Multinomial_recall),3)*100, round(sum(Gaussian_recall) / len(Gaussian_recall),3)*100, round(sum(Bernuilli_recall) / len(Bernuilli_recall),3)*100]
plt.bar(eje_x, eje_y, color=['blue','red','green'])
plt.ylabel('Recall')
plt.xlabel('Modelo')
plt.title('Naive Bayes')
plt.show()


# %%
eje_y = [ round(sum(Multinomial_precision) / len(Multinomial_precision),3)*100, round(sum(Gaussian_precision) / len(Gaussian_precision),3)*100, round(sum(Bernuilli_precision) / len(Bernuilli_precision),3)*100]
plt.bar(eje_x, eje_y, color=['blue','red','green'])
plt.ylabel('Precision')
plt.xlabel('Modelo')
plt.title('Naive Bayes')
plt.show()

# %%
eje_y = [ round(sum(Multinomial_f1) / len(Multinomial_f1),3)*100, round(sum(Gaussian_f1) / len(Gaussian_f1),3)*100, round(sum(Bernuilli_f1) / len(Bernuilli_f1),3)*100]
plt.bar(eje_x, eje_y, color=['blue','red','green'])
plt.ylabel('F1')
plt.xlabel('Modelo')
plt.title('Naive Bayes')
plt.show()


