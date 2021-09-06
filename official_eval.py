
import sys
import json

from sklearn.metrics import classification_report


def main():
    if len(sys.argv) < 4:
        print("Usage")
        print("\tpython {0:s} gt_json pred_json simple".format(sys.argv[0]))
        print("Where")
        print("\tgt_json\tDataset Ground Truth labels in JSON")
        print("\tpred_json\tDataset Predicted labels in JSON")
        print("\tsimple\t\tDetermines classification type")
        print("\t\t\t1 for bill type only")
        print("\t\t\t0 for type+side classification")
        return

    gt_filename = sys.argv[1]
    pred_filename = sys.argv[2]
    try:
        class_simple = int(sys.argv[3]) > 0
    except:
        print("valor simple es invalido")
        return    

    print("\n")
    print("Archivo GT: " + gt_filename)
    print("Archivo JSON: " + pred_filename)
    if class_simple:
        print("Clasificacion Simple (tipo de billete solamente)")
    else:
        print("Clasificacion Completa (tipo de billete + Cara)")
    print("\n")

    with open(gt_filename, "r") as in_json:
        gt = json.load(in_json)
        
    with open(pred_filename, "r") as in_json:
        pred = json.load(in_json)

    y_gt = []
    y_pred = []
    for key in gt:
        if class_simple:
            y_gt.append(gt[key]["denominacion"])
        else:
            y_gt.append("{0:s}_{1:s}".format(gt[key]["denominacion"], gt[key]["lado"]))

        if key not in pred:
            y_pred.append("-1")
        else:
            if class_simple:
                if "denominacion" in pred[key]:
                    y_pred.append(pred[key]["denominacion"])
                else:
                    raise Exception("No se esperaba: " + key + ": " + str(pred[key]))
            else:
                if "denominacion" in pred[key] and "lado" in pred[key]:
                    y_pred.append("{0:s}_{1:s}".format(pred[key]["denominacion"], pred[key]["lado"]))
                else:
                    raise Exception("No se esperaba: " + key + ": " + str(pred[key]))                    

    print(classification_report(y_gt, y_pred, digits=6, zero_division=0))

if __name__ == "__main__":
    main()

