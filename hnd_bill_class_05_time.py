import os
import sys
import time
import cv2
import numpy
import joblib
from hnd_bill_class_00_preprocesamiento import process_file
from hnd_bill_class_01_features import get_descriptors_plants, get_plantillas
from hnd_bill_class_01_features import det_algo_type
from hnd_bill_class_01_features import get_keypoints_ORB
from hnd_bill_class_01_features import get_keypoints_SIFT
from hnd_bill_class_01_features import match_plantilla_ORB
from hnd_bill_class_01_features import match_plantilla_SIFT
from sklearn.preprocessing import StandardScaler


def main():
    print("hola")
    if len(sys.argv) < 5:
        print("Uso:")
        print("img_dir dir_plantillas preproc[yes/no] algo_type .joblib")
        return

    in_dir = sys.argv[1]
    dir_plantillas = sys.argv[2]
    preproc_yes_no = sys.argv[3]
    algo_type = sys.argv[4]
    class_joblib = sys.argv[5]

    #root_path, filename = os.path.split(in_dir)

    start_time = time.time()
    img = None
    if preproc_yes_no == 'YES' or preproc_yes_no == 'yes':
        recorte_w_bg, full_no_bg, recorte_no_bg, in_path = process_file(in_dir)
        img = recorte_no_bg
        print("1. prepoc")
    elif preproc_yes_no == 'NO' or preproc_yes_no == 'no':
        img = cv2.imread(in_dir)
        print("1. no preproc")

    algo_type = det_algo_type(algo_type)
    plantillas = get_plantillas(dir_plantillas)

    #Descriptors de Plantillas
    print("2. Descriptors temps")
    descriptors_temps = get_descriptors_plants(algo_type,plantillas)
    #Keypoints imagen
    print("3. Descriptor imagen")
    descriptor = get_keypoints(img,algo_type)
    #Matching with Templates
    print("4. Matching Temps")
    matches_bills = template_matching(algo_type,descriptor,descriptors_temps)    
    features = list()
    #scaler = StandardScaler()
    #print(type(matches_bills))
    #x = scaler.fit_transform([matches_bills])
    #print(type(x),"--------",x)
    #features.append(['1_f','1_r','2_f','2_r','5_f','5_r',
    #         '10_f','10_r','20_f','20_r','50_f','50_r',
    #         '100_f','100_r','200_f','200_r','500_f','500_r'])
    #features.append(matches_bills)
    #print(features)
    print("5. Predicting")
    scaler, trained_classifier = joblib.load(class_joblib)

    x = scaler.transform([matches_bills])
    pred = trained_classifier.predict(x)
    print('Prediccion: ', pred)

    end_time = time.time()
    print("Tiempo: ", (end_time - start_time), " segundos")

def get_keypoints(imagen,algo_type):
    if algo_type == 'ORB':
        return get_keypoints_ORB(imagen)
    elif algo_type == 'SIFT':
        return get_keypoints_SIFT(imagen)

def template_matching(algo_type,descriptor,descriptors_temps ):
    matches_bills = list()
    #ls_desc_billete = [descriptor] * len(descriptors_temps)
    for desc_temp in descriptors_temps:
        if algo_type == 'ORB':
            good_matches = match_plantilla_ORB(descriptor, desc_temp)
            matches_bills.append(good_matches)
                                            
        elif algo_type == 'SIFT':                                
            good_matches = match_plantilla_SIFT(descriptor, desc_temp)
            matches_bills.append(good_matches)

    #Normalizando keypoints
    total_bill = sum(matches_bills)
    for t in range(len(matches_bills)):
        if total_bill > 0:
            matches_bills[t] = round(matches_bills[t] / total_bill,5)    

    return matches_bills                

if __name__ == '__main__':
    main()