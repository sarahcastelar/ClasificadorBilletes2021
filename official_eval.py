
import sys
import json

from sklearn.metrics import classification_report


def main():
    if len(sys.argv) < 3:
        print("Usage")
        print("\tpython {0:s} gt_json pred_json".format(sys.argv[0]))
        return

    gt_filename = sys.argv[1]
    pred_filename = sys.argv[2]

    with open(gt_filename, "r") as in_json:
        gt = json.load(in_json)
        
    with open(pred_filename, "r") as in_json:
        pred = json.load(in_json)

    y_gt = []
    y_pred = []
    for key in gt:
        y_gt.append(gt[key]["denominacion"])

        if key not in pred:
            y_pred.append("-1")
        else:
            if "denominacion" in pred[key]:
                y_pred.append(pred[key]["denominacion"])
            else:
                if isinstance(pred[key], str):
                    y_pred.append(pred[key])
                else:
                    raise Exception("No se esperaba: " + key + ": " + str(pred[key]))

    print(classification_report(y_gt, y_pred, digits=6))

if __name__ == "__main__":
    main()

