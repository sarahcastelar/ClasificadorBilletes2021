import os
import sys
import time
import cv2
from hnd_bill_class_00_preprocesamiento import process_file
from hnd_bill_class_01_features import get_descriptors_plants, get_keypoints, get_plantillas
from hnd_bill_class_01_features import det_algo_type
from hnd_bill_class_01_features import get_keypoints_ORB
from hnd_bill_class_01_features import get_keypoints_SIFT
from hnd_bill_class_01_features import match_plantilla_ORB
from hnd_bill_class_01_features import match_plantilla_SIFT

def main():
    print("hola")
    if len(sys.argv) < 5:
        print("Uso:")
        print("img_dir dir_plantillas preproc[yes/no] algo_type")
        return

    in_dir = sys.argv[1]
    dir_plantillas = sys.argv[2]
    preproc_yes_no = sys.argv[3]
    algo_type = sys.argv[4]

    root_path, filename = os.path.split(in_dir)

    start_time = time.time()
    img = None
    if preproc_yes_no == 'YES' or preproc_yes_no == 'yes':
        recorte_w_bg, full_no_bg, recorte_no_bg, in_path = process_file(in_dir)
        img = recorte_no_bg
        print("prepoc")
    elif preproc_yes_no == 'NO' or preproc_yes_no == 'no':
        img = cv2.imread(in_dir)
        print("no preproc")

    algo_type = det_algo_type(algo_type)
    plantillas = get_plantillas(dir_plantillas)

    #Descriptors de Plantillas
    print("descriptors temps")
    descriptors_temps = get_descriptors_plants(algo_type,plantillas)
    #Keypoints imagen
    descriptor = get_keypoints(img,algo_type)
    #Matching with Templates
    matches_bills = template_matching(algo_type,descriptor,descriptors_temps)    
    matches_bills.append(filename)
    print(matches_bills)

    end_time = time.time()
    print("Tiempo: ", (end_time - start_time), " segundos")

def get_keypoints(imagen,algo_type):
    if algo_type == 'ORB':
        return get_keypoints_ORB(imagen)
    elif algo_type == 'SIFT':
        return get_keypoints_SIFT(imagen)

def template_matching(algo_type,descriptor,descriptors_temps ):
    matches_bills = list()
    ls_desc_billete = [descriptor] * len(descriptors_temps)
    
    if algo_type == 'ORB':
        good_matches = match_plantilla_ORB(ls_desc_billete, descriptors_temps)
        matches_bills.append(good_matches)
                                        
    elif algo_type == 'SIFT':                                
        good_matches = match_plantilla_SIFT(ls_desc_billete, descriptors_temps)
        matches_bills.append(good_matches)

    #Normalizando keypoints
    total_bill = sum(matches_bills)
    for t in range(len(matches_bills)):
        if total_bill > 0:
            matches_bills[t] = round(matches_bills[t] / total_bill,5)    

    return matches_bills                
        