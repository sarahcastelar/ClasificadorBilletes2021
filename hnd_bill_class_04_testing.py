#mp2_04_evaluar_clasificador.py

#Ejemplo de como correrlo:
#& python mp2_04_evaluar_clasificador.py caracs.csv clasificador.joblib out_etiqs.json

import cv2 as cv
import joblib
from numpy import *
import numpy
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore') 
import json
import re

def Testing(datos,trained_classifier,output):
    x = datos.loc[:,datos.columns!='ID']
    name_files = datos['ID']

    clf = trained_classifier
    pred = clf.predict(x)
    data = {}
    for i in range(len(name_files)):
        data[name_files[i]] = {}

        #Get lado de la prediccion
        lado = re.findall(r'(f|r)',pred[i])
        if lado[0] == 'f':
            lado = 'frontal'
        elif lado[0] == 'r':
            lado = 'reverso'

        #Get denominacion de la prediccion
        denominacion = (re.findall(r'([0-9]+)',pred[i]))[0]
        denom_y = {"denominacion" : denominacion,
                "lado" : lado}

        #print(denom_y,pred[i])
        data[name_files[i]].update(denom_y)

    #Creacion de Json
    file = open(output,'w')
    json.dump(data,file,indent=4)
    file.close()
    print("--------------------------------------")
    print("hnd_bill_class_04_testing.py completado con exito")
    print("Archivo de output creado y guardado llamado: ",output)
    print("--------------------------------------")


def main():
    classifier = sys.argv[2]
    output = sys.argv[3]
    if classifier.__contains__('.joblib') and output.__contains__('.json'):
        datos_csv = pd.read_csv(sys.argv[1])
        trained_classifier = joblib.load(sys.argv[2])
        Testing(datos_csv,trained_classifier,output)
    else:
        print("Archivo de entrada debe contener el formato .joblib al final")
        print("Archivo de output en formato json necesita .json al final")


if __name__ == '__main__':
    main()