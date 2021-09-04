#USO:
#& python .\hnd_bill_class_01_features.py 
# .\preproc
# caracs.csv sift

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


def main():
    features = []
    plantillas = []

    if len(sys.argv) < 4:
        print('Missing Directory or File Name!!')
        return
    directory = sys.argv[1]
    file_name = sys.argv[2]
    algo_type = sys.argv[3]

    if algo_type == 'orb' or algo_type == 'ORB':
        algo_type = 'ORB'
    elif algo_type == 'sift' or algo_type == 'SIFT':
        algo_type = 'SIFT'
    else:
        print("Debe elegir ORB o SIFT")
        exit()

    # directory = "dir"
    start = time.time()

    plantillas = get_plantillas()    
    print("Por favor espere.. se está analizando. Puede tardar.")
    
    indice = 0
    file_names =  os.listdir(directory)
    path = directory + '/*.jpg'
    imagenes = [cv.imread(file) for file in glob.glob(path)]

    for imagen in imagenes:
        print("En imagen: ",file_names[indice])
        print("Extrayendo keypoints")
        row_keypoints = list()
        row_keypoints = get_keypoints(algo_type,imagen,plantillas)
        print("Extrayendo RGB features")
        # features
        row_features = getting_color_values(imagen)
        row_keypoints.append(file_names[indice])
        new_row = row_features + row_keypoints
        features.append(new_row)

        indice+=1

    # print(row_bgr)
    # cv.waitKey(0)
    # print(features)
    print("Escribiendo CSV")
    write_csv(features, file_name)
    end = time.time()
    print("Tiempo: ", (end - start), " segundos")
    return

def get_keypoints(algo_type,imagen,plantillas):
    if algo_type == 'ORB':
        return get_keypoints_ORB(imagen,plantillas)
    elif algo_type == 'SIFT':
        return get_keypoints_SIFT(imagen,plantillas)

def get_keypoints_SIFT(imagen, plantillas):
    row_keypoints = []
    class_id = 0
    for plantilla in plantillas:
        good_matches = 0
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(imagen, None)
        kp2, des2 = sift.detectAndCompute(plantilla, None)
        # FLANN parameters
        index = dict(algorithm=1, trees=5)
        search = dict(checks=50)
        flann = cv.FlannBasedMatcher(index, search)
        matches = flann.knnMatch(des1, des2, k=2)
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good_matches += 1
        row_keypoints.append(good_matches)
    return row_keypoints

def get_keypoints_ORB(imagen,plantillas):
    roi_gray = cv.cvtColor(imagen,cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(roi_gray,None)
    bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck = True)

    matches_bills = list()
    for temp in plantillas:
        kp2, des2 = orb.detectAndCompute(temp,None)

        matches = bf.match(des1,des2)
        matches = sorted(matches,key = lambda x:x.distance)

        num_matches = len(matches)
        matches_bills.append(num_matches)
    
    #Normalizando keypoints
    total_bill = sum(matches_bills)
    for t in range(len(matches_bills)):
        if total_bill > 0:
            matches_bills[t] = round(matches_bills[t] / total_bill,5)
    
    return matches_bills

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
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    for (percent, color) in zip(hist, clt.cluster_centers_):  # BGR
        b = round(((color[0] / 255) - 0.406) / 0.225,3)
        g = round(((color[1] / 255) - 0.456) / 0.224,3)
        r = round(((color[2] / 255) - 0.485) / 0.229,3)
        colores.append(b)
        colores.append(g)
        colores.append(r)
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
                     color.astype("uint8").tolist(), -1)
        startX = endX
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


def get_plantillas():
    path = './nuevas_plantillas/*.jpg'
    images = [cv.imread(file) for file in glob.glob(path)]
    return images


if __name__ == '__main__':
    main()
