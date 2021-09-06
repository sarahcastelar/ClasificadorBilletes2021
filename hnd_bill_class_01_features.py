#USO:
#& python .\hnd_bill_class_01_features.py 
# .\preproc (carpeta de imagenes preprocesadas)
# caracs1.csv (nombre del csv output)
# sift (algo a usar)

from cv2 import data
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans, vq
import time
import pandas as pd
import csv
import os
import sys
import glob
from concurrent.futures import ProcessPoolExecutor


def match_plantilla_SIFT(des_billete, desc_plantilla):
    index = dict(algorithm=1, trees=5)
    search = dict(checks=50)
    flann = cv.FlannBasedMatcher(index, search)
    
    matches = flann.knnMatch(des_billete, desc_plantilla, k=2)

    good_matches = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good_matches += 1
                            
    return good_matches


def match_plantilla_ORB(des_billete, desc_plantilla):
    bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck = True)

    matches = bf.match(des_billete, desc_plantilla)                    
    # matches = sorted(matches,key = lambda x:x.distance)
    good_matches = [x for x in matches if x.distance < 64]

    print(f"Raw matches: {len(matches)}, Good matches: {len(good_matches)}")
                                        
    num_matches = len(good_matches)

    return num_matches


def main():
    features = list()
    plantillas = list()

    if len(sys.argv) < 4:
        print("Usage: ")
        print(f"\tpython {sys.argv[0]} img_dir out_csv algor plant_dir")
        print("With:")
        print("\timg_dir:\tInput image directory")
        print("\tout_csv:\tOutput CSV file")
        print("\talgorithm:\tSIFT or ORB")
        print("\tplant_dir:\tTemplate Directory")
        return
    
    directory = sys.argv[1]
    file_name = sys.argv[2]
    algo_type = sys.argv[3]
    dir_plant = sys.argv[4]

    algo_type = det_algo_type(algo_type)

    # directory = "dir"
    start = time.time()

    plantillas = get_plantillas(dir_plant)    
    print("Por favor espere.. se está analizando. Puede tardar.")
    
    indice = 0
    file_names =  os.listdir(directory)
    file_paths = [directory + '/' + n for n in file_names]
    # file_paths = file_paths[:10]
    #path = directory + '/*.jpg'
    
    #imagenes = [cv.imread(file) for file in glob.glob(path)]
    #print(file_paths)
    #print(path)
    #exit()

    print("... procesando plantillas ... ")

    #Descriptors de Plantillas
    descriptors_temps = list()
    if algo_type == 'ORB':
        for temp in plantillas:
            orb = cv.ORB_create(nfeatures=1000)
            kp2, des2 = orb.detectAndCompute(temp,None)
            descriptors_temps.append(des2)
    elif algo_type == 'SIFT':
        for temp in plantillas:    
            sift = cv.SIFT_create()
            kp2, des2 = sift.detectAndCompute(temp, None)
            descriptors_temps.append(des2)    

    #Parelizacion de Features del Dataset (keypoints y rgbs)
    with ProcessPoolExecutor(max_workers=10) as executor:
        print("... procesando imagenes ... ")
        all_results = []
        for imagen_path, descriptor, color_features in executor.map(get_keypoints, file_paths):
            print("- completado: ", imagen_path)

            all_results.append((imagen_path, descriptor, color_features))
            
        print("... haciendo template matching ... ")
        for imagen_path, descriptor, color_features in all_results:
            root_path, filename = os.path.split(imagen_path)
            
            st = time.time()

            print("- procesando: " + imagen_path)
            
            matches_bills = list()
            ls_desc_billete = [descriptor] * len(descriptors_temps)
            
            if algo_type == 'ORB':
                for good_matches in executor.map(match_plantilla_ORB, ls_desc_billete, descriptors_temps):
                    matches_bills.append(good_matches)
                                                
            elif algo_type == 'SIFT':                                
                for good_matches in executor.map(match_plantilla_SIFT, ls_desc_billete, descriptors_temps):
                    matches_bills.append(good_matches)

            #Normalizando keypoints
            total_bill = sum(matches_bills)
            for t in range(len(matches_bills)):
                if total_bill > 0:
                    matches_bills[t] = round(matches_bills[t] / total_bill,5)                    
                                    
            end = time.time()
            print("tiempo: ", end - st)
                
            matches_bills.append(filename)
            row = color_features + matches_bills
            features.append(row)
            
            
    print("Escribiendo CSV")
    print(features[0])
    write_csv(features, file_name)
    end = time.time()
    print("Tiempo: ", (end - start), " segundos")
    return

def get_keypoints(imagen_path):
    algo_type = det_algo_type(sys.argv[3])
    imagen = cv.imread(imagen_path)

    if algo_type == 'ORB':
        return imagen_path,get_keypoints_ORB(imagen),getting_color_values(imagen)
    elif algo_type == 'SIFT':
        return imagen_path,get_keypoints_SIFT(imagen),getting_color_values(imagen)



def get_keypoints_SIFT(imagen):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(imagen, None)
    return des1

def get_keypoints_ORB(imagen):
    roi_gray = cv.cvtColor(imagen,cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(roi_gray,None)
    return des1

def getting_color_values(image):
    colores = list()
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=7)
    clt.fit(image)
    # .reshape(1, -1)
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    #bar = np.zeros((50, 300, 3), dtype="uint8")
    #startX = 0
    for (percent, color) in zip(hist, clt.cluster_centers_):  # BGR
        b = round(((color[0] / 255) - 0.406) / 0.225,3)
        g = round(((color[1] / 255) - 0.456) / 0.224,3)
        r = round(((color[2] / 255) - 0.485) / 0.229,3)
        colores.append(b)
        colores.append(g)
        colores.append(r)
        #endX = startX + (percent * 300)
        #cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
        #             color.astype("uint8").tolist(), -1)
        #startX = endX
    # # print(final_colors)
    # cv.imshow("palette", bar)
    # cv.waitKey(0)

    return colores

def write_csv(rows, output):
    cols = ['B0', 'G0', 'R0', 'B1', 'G1', 'R1', 'B2', 'G2', 'R2',
            'B3', 'G3', 'R3', 'B4', 'G4', 'R4', 'B5', 'G5', 'R5',
            'B6', 'G6', 'R6', '1_f','1_r','2_f','2_r','5_f','5_r','10_f','10_r',
            '20_f','20_r','50_f','50_r','100_f','100_r','200_f','200_r','500_f','500_r','ID']
    
    with open(output, 'w', newline='', encoding='UTF8') as file:
        writer = csv.writer(file)
        writer.writerow(cols)
        writer.writerows(rows)


def get_plantillas(dir_path):
    path = dir_path + '/*.jpg'
    all_paths = []
    for sub_path in glob.glob(path):
        path, filename = os.path.split(sub_path)

        parts = filename.split("_")
        num = int(parts[0])
        face = parts[1][0]

        all_paths.append((num, face, sub_path))

    all_paths = sorted(all_paths)

    images = [cv.imread(file) for _, _, file in all_paths]

    return images

def det_algo_type(algo_type):
    if algo_type == 'orb' or algo_type == 'ORB':
        algo_type = 'ORB'
        return algo_type
    elif algo_type == 'sift' or algo_type == 'SIFT':
        algo_type = 'SIFT'
        return algo_type
    else:
        print("Debe elegir ORB o SIFT")
        exit()

if __name__ == '__main__':
    main()
