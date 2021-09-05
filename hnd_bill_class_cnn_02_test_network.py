import torch
from torch.utils.data import DataLoader
import sys
import json
import os
import numpy as np
from billetes_dataset import BilletesDataset
from Lempira import LempiraNet


def main():
    if len(sys.argv) < 4:
        print("Uso:")
        print(f"\tpython {sys.argv[0]} in_model in_dir out_json")
        return
    in_model = sys.argv[1]
    in_dir = sys.argv[2]
    out_json = sys.argv[3]

    data = BilletesDataset(root_dir=in_dir, etiquetas={}, size=(320, 128))
    model = LempiraNet(ratio_width=5, ratio_height=2)
    model.load_state_dict(torch.load(in_model))
    model.eval()
    loader = DataLoader(data, batch_size=1, shuffle=False)
    classes = [
        "1-frontal",
        "1-reverso",
        "2-frontal",
        "2-reverso",
        "5-frontal",
        "5-reverso",
        "10-frontal",
        "10-reverso",
        "20-frontal",
        "20-reverso",
        "50-frontal",
        "50-reverso",
        "100-frontal",
        "100-reverso",
        "200-frontal",
        "200-reverso",
        "500-frontal",
        "500-reverso"
    ]

    output_dict = {}
    for (x, name) in loader:
        img_name = name[0]
        output_dict[img_name] = {}
        full_predict = classes[np.argmax(
            model(x).detach().numpy())]
        denominacion = full_predict.split('-')[0]
        lado = full_predict.split('-')[1]
        output_dict[img_name]["denominacion"] = denominacion
        output_dict[img_name]["lado"] = lado

    json.dump(output_dict, open(out_json, 'w'), indent=4)


if __name__ == '__main__':
    main()