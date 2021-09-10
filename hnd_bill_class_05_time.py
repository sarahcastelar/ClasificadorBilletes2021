import os
import sys
import time
import cv2
from hnd_bill_class_00_preprocesamiento import process_file
from hnd_bill_class_01_features import get_descriptors_plants, get_keypoints, get_plantillas
from hnd_bill_class_01_features import det_algo_type
from hnd_bill_class_01_features import get_keypoints_ORB
from hnd_bill_class_01_features import get_keypoints_SIFT
from hnd_bill_class_01_features import getting_color_values
from hnd_bill_class_01_features import match_plantilla_ORB
from hnd_bill_class_01_features import match_plantilla_SIFT

def main():
    if len(sys.argv) < 5:
        print("Uso:")
        print("\tpython {0:s} in_dir out_dir".format(sys.argv[0]))
        return

    in_dir = sys.argv[1]
    dir_plant = sys.argv[2]
    preproc_yes_no = sys.argv[3]
    algo_type = sys.argv[4]


    #all_paths = [in_dir + "/" + name for name in os.listdir(in_dir)]

    start_time = time.time()
    img = None
    if preproc_yes_no == 'YES' or preproc_yes_no == 'YES':
        recorte_w_bg, full_no_bg, recorte_no_bg, in_path = process_file(in_dir)
        root_path, filename = os.path.split(in_path)
        img = recorte_no_bg
        #out_path = out_dir + "/" + filename
        #print(out_path)
        #cv2.imwrite(out_path, recorte_no_bg)
    elif preproc_yes_no == 'NO' or preproc_yes_no == 'no':
        img = cv2.imread(in_dir)

    algo_type = det_algo_type(algo_type)
    plantillas = get_plantillas(dir_plant)

    #Descriptors de Plantillas
    descriptors_temps = get_descriptors_plants(algo_type,plantillas)
    #Keypoints imagen
    descriptor, color_features = get_keypoints(img,algo_type)

    matches_bills = list()
    template_matching()

    end_time = time.time()
    print("Tiempo: ", (end_time - start_time), " segundos")

def get_keypoints(imagen,algo_type):
    if algo_type == 'ORB':
        return get_keypoints_ORB(imagen),getting_color_values(imagen)
    elif algo_type == 'SIFT':
        return get_keypoints_SIFT(imagen),getting_color_values(imagen)

def template_matching(algo_type,descriptor,descriptors_temps ):
    matches_bills = list()
    ls_desc_billete = [descriptor] * len(descriptors_temps)
    
    if algo_type == 'ORB':
        good_matches = match_plantilla_ORB(ls_desc_billete, descriptors_temps)
        matches_bills.append(good_matches)
                                        
    elif algo_type == 'SIFT':                                
        for good_matches in executor.map(match_plantilla_SIFT, ls_desc_billete, descriptors_temps):
            matches_bills.append(good_matches)

    #Normalizando keypoints
    total_bill = sum(matches_bills)
    for t in range(len(matches_bills)):
        if total_bill > 0:
            matches_bills[t] = round(matches_bills[t] / total_bill,5)                    
        