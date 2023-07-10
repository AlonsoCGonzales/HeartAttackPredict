#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:42:51 2023

@author: tori
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc


data = pd.read_csv('./heart.csv')
#numeric_columns = data.select_dtypes(include=['int64', 'float64'])

scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

data = normalized_data

data.head()
data.info()
data.describe()

data_sorted = data.sort_values('age')
print(data_sorted)

# Limpiamos el dataset
Q1 = normalized_data.quantile(0.25)
Q3 = normalized_data.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

filtered_data = normalized_data[~((normalized_data < lower_bound) | (normalized_data > upper_bound)).any(axis=1)]

print(filtered_data)
"""

|age|sex|cp|trtbps|chol|fbs|restecg|thalachh|exng|oldpeak|slp|caa|thall|output|

"output"-> variable dependiente

"""

# Separar características y variable objetivo "output"
X = data.iloc[:, :-1]
y = data['output']

"""
print(X)
"""


# Division de datos en sets de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y ajustar el modelo de Logistic Regression
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# Crear y ajustar el modelo de Linear Discriminant Analysis
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)

# Predecir con los modelos
y_pred_logreg = logreg_model.predict(X_test)
y_pred_lda = lda_model.predict(X_test)

"""
Medir la eficiencia (ejemplo: Accurracy, cross validation, curvas de ROC
Clasification report, MATRIZ DE CONFUSION )

"""

# Calculo del Accurrady de los modelos
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
accuracy_lda = accuracy_score(y_test, y_pred_lda)

# Imprimir el resultado del Accurracy
print("\nAccurracy (Logistic Regression):", accuracy_logreg)
print("Accurracy (Linear Discriminant Analysis):", accuracy_lda)

# Obtener el cross validation
crossval_logreg = cross_val_score(logreg_model, X_train, y_train, cv=5)
crossval_lda = cross_val_score(lda_model, X_train, y_train, cv=5)

# Imprimir la prueba de Cross Validation
print("\nCross Validation (Logistic Regression):")
print(crossval_logreg)
print("Cross Validation (Linear Discriminant Analysis):")
print(crossval_lda)

# Obtener el Clasification report
classification_report_logreg = classification_report(y_test, y_pred_logreg)
classification_report_lda = classification_report(y_test, y_pred_lda)

# Imprimir Clasification report
print("\nClasification report (Logistic Regression):")
print(classification_report_logreg)
print("Clasification report (Linear Discriminant Analysis):")
print(classification_report_lda)


# Obtener la Matriz de Confusion de ambos algritmos
confusion_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)
confusion_matrix_lda = confusion_matrix(y_test, y_pred_lda)

# Imprimir la matriz de confusión
print("\nConfussion matrix (Logistic Regression):")
print(confusion_matrix_logreg)
print("Confussion matrix (Linear Discriminant Analysis):")
print(confusion_matrix_lda)

# Calcular las curvas ROC y el Area bajo la curva (AUC)
y_pred_prob_logreg = logreg_model.predict_proba(X_test)[:, 1]
fpr_logreg, tpr_logreg, thresholds_logreg = roc_curve(y_test, y_pred_prob_logreg)
roc_auc_logreg = auc(fpr_logreg, tpr_logreg)

y_pred_prob_lda = lda_model.predict_proba(X_test)[:, 1]
fpr_lda, tpr_lda, thresholds_lda = roc_curve(y_test, y_pred_prob_lda)
roc_auc_lda = auc(fpr_lda, tpr_lda)

# Imprimir el Area bajo la curva (AUC) de LR y LDA
print("\nAUC (Logistic Regression):", roc_auc_logreg)
print("AUC (Linear Discriminant Analysis):", roc_auc_lda)

### Graficas 9 ya para que no jodas

# Histograma de la edad
plt.hist(data['age'], bins=10)
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.title('Distribucion de la edad')
plt.show()

# Diagrama de dispersión de edad y colesterol
plt.scatter(data['age'], data['chol'])
plt.xlabel('Edad')
plt.ylabel('Colesterol')
plt.title('Relación entre edad y colesterol')
plt.show()


# Grafico de pastel genero
data['sex'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Proporcion Demografica')
plt.show()


# Grafico de densidad de la presión arterial en reposo
sns.kdeplot(data['trtbps'], shade=True)
plt.xlabel('Presion arterial en reposo')
plt.ylabel('Densidad')
plt.title('Distribucion de la presión arterial en reposo')
plt.show()

# Grafico de Cajas de variables normalizadas

plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_data)
plt.title('Grafico de Cajas de Variables Numericas Normalizadas')
plt.xticks(rotation=45)
plt.show()

# Comparativa Curvas Roc
plt.figure()
plt.plot(fpr_logreg, tpr_logreg, label='Logistic Regression (AUC = {:.2f})'.format(roc_auc_logreg))
plt.plot(fpr_lda, tpr_lda, label='Linear Discriminant Analysis (AUC = {:.2f})'.format(roc_auc_lda))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Tasa de falsos positivos (FPR)')
plt.ylabel('Tasa de verdaderos positivos (TPR)')
plt.title('Curvas ROC')
plt.legend(loc='lower right')
plt.show()

