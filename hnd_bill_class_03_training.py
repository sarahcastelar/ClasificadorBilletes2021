
#Ejemplo de como correrlo:

#& python mp2_03_entrenar_clasificador.py caracs.csv .\training_image_dataset\training_gt.json clasificador.joblib RF

import cv2 as cv
import joblib
from numpy import *
import numpy
import sys
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore') 
import json
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
from sklearn.svm import SVC


def RFC(datos,y,output):
    x = datos.loc[:,datos.columns!='ID']
    clf = RandomForestClassifier(criterion='gini')
    clf.n_estimators = 55
    clf.max_depth = 130
    clf.max_features = 17

    x_train = x
    y_train = y

    clf.fit(x_train,y_train)
    pred = clf.predict(x_train)
    reporte = classification_report(y_train, pred,labels=["1_f", "1_r", "2_f", "2_r", "5_f", "5_r", 
        "10_f", "10_r", "20_f", "20_r", "50_f", "50_r",  "100_f", "100_r", "200_f", "200_r","500_f","500_r"]) 

    print("-----------REPORT with Fit")
    print(reporte)
    print("Matriz de Confusion: ")
    print(metrics.confusion_matrix(y_train,pred))
    print("Accu: ", round(metrics.accuracy_score(y_train,pred),4))

    joblib.dump(clf,output)
    print("Clasificador guardado como: ", output)

def SVM(datos_csv, y, output):

    svc = SVC(kernel='rbf', gamma=0.25, C=35)
    svc.fit(datos_csv.loc[:, datos_csv.columns !='ID'], y)

    y_predict = svc.predict(datos_csv.loc[:, datos_csv.columns !='ID'])

    # exportar modelo...
    joblib.dump(svc, output)
    print("Clasificador exportado en: ", output)

    # print(y_predict)
    f1_score = metrics.f1_score(y, y_predict, average='weighted')

    accuracy = metrics.accuracy_score(y, y_predict)

    report = metrics.classification_report(y, y_predict, labels=[
        "1_f", "1_r", "2_f", "2_r", "5_f", "5_r", "10_f", "10_r", "20_f", "20_r", 
        "50_f", "50_r",  "100_f", "100_r", "200_f", "200_r","500_f","500_r"])

    cm = metrics.confusion_matrix(y, y_predict)

    print("--------------------------PROMEDIOS------------------------------")
    print("F1 Score: ", round(f1_score, 4))
    print("Accuracy: ", round(accuracy, 4))
    print("--------------------Classification Report-----------------------")
    print(report)
    print("---------------------Matriz de Confusión------------------------")
    print("\n", cm)
    print("----------------------------------------------------------------")


def main():
    if len(sys.argv) < 5:
        print("Hacen falta argumentos.")
        print("Ejemplo: & python mp2_03_entrenar_clasificador.py caracs.csv training_gt.json clasificador.joblib RF")
        return
    output = sys.argv[3]
    algo = sys.argv[4]
    if output.__contains__('.joblib'):
        #Cargar csv de features
        datos_csv = pd.read_csv(sys.argv[1])

        values = list()
        #Cargar json de clases
        f = open(sys.argv[2])
        datos_json = json.load(f)

        for imagen,datos in datos_json.items():
            if datos['lado'] == 'frontal':
                clase = datos['denominacion'] + '_' + 'f'
            elif datos['lado'] == 'reverso': 
                clase = datos['denominacion'] + '_' + 'r'
            values.append(clase)
        
        #Y para predecir
        y = pd.Series(values)
        #print(y,"----")

        #Nombre del archivo de output
        output = sys.argv[3]
        #Entrenamiento
        if algo == 'SVM' or algo == 'svm':
            SVM(datos_csv,y,output)
        elif algo == 'RF' or algo == 'rf':
            RFC(datos_csv,y,output)
    else:
        print("Archivo de salida debe contener el formato .joblib al final")


if __name__ == '__main__':
    main()