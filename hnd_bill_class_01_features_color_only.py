
import os
import sys
import time
import csv

from concurrent.futures import ProcessPoolExecutor

import cv2

from hnd_bill_class_color_features import getting_color_values_v2

def get_color_features(imagen_path):
    imagen = cv2.imread(imagen_path)

    return imagen_path, getting_color_values_v2(imagen)


def write_csv(rows, output):
    cols = ['B0', 'G0', 'R0', 'B1', 'G1', 'R1', 'B2', 'G2', 'R2',
            'B3', 'G3', 'R3', 'B4', 'G4', 'R4', 'B5', 'G5', 'R5',
            'B6', 'G6', 'R6', 'ID']
    
    with open(output, 'w', newline='', encoding='UTF8') as file:
        writer = csv.writer(file)
        writer.writerow(cols)
        writer.writerows(rows)


def main():
    if len(sys.argv) < 3:
        print("Usage: ")
        print(f"\tpython {sys.argv[0]} img_dir out_csv")
        print("With:")
        print("\timg_dir:\tInput image directory")
        print("\tout_csv:\tOutput CSV file")
        return
    
    directory = sys.argv[1]
    file_name = sys.argv[2]
    
    # directory = "dir"
    start = time.time()

    indice = 0
    file_names =  os.listdir(directory)
    file_paths = [directory + '/' + n for n in file_names]

    features = []
    
    #Parelizacion de Features del Dataset (rgbs only)
    with ProcessPoolExecutor(max_workers=10) as executor:
        print("... procesando imagenes ... ")
        
        for imagen_path, color_features in executor.map(get_color_features, file_paths):
            print("- completado: ", imagen_path)

            root_path, filename = os.path.split(imagen_path)

            row = color_features
            row.append(filename)
            
            features.append(row)
            
    print("Escribiendo CSV")
    print(features[0])
    write_csv(features, file_name)
    end = time.time()
    print("Tiempo: ", (end - start), " segundos")
    return


if __name__ == "__main__":
    main()

