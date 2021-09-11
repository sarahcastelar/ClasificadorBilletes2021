import torch
import torchvision.transforms.functional as TF
import sys
import numpy as np
from hnd_bill_class_00_preprocesamiento import process_file
from PIL import Image
from Lempira import LempiraNet


def add_padding(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def safe_resize(pil_img, size):
    img_width, img_height = pil_img.size
    target_width, target_height = size

    scale_w = target_width/img_width
    scale_h = target_height/img_height

    factor = 0
    if scale_h >= scale_w:
        factor = scale_w
        pil_img = pil_img.resize(
            (target_width, int(pil_img.height * factor)))
        diff = (target_height - pil_img.height)
        padding_top = diff // 2
        padding_bottom = diff - padding_top
        pil_img = add_padding(
            pil_img, padding_top, 0, padding_bottom, 0, (0, 0, 0))
    else:
        factor = scale_h
        pil_img = pil_img.resize(
            (int(pil_img.width * factor), target_height))
        diff = (target_width - pil_img.width)
        padding_right = diff // 2
        padding_left = diff - padding_right
        pil_img = add_padding(
            pil_img, 0, padding_right, 0, padding_left, (0, 0, 0))
    return pil_img


def main():
    if len(sys.argv) < 4:
        print("Uso:")
        print(f"\tpython {sys.argv[0]} in_model img preproccess cuda")
        return
    accepted = ["True", "true", "enable", "yes", "Yes", "1"]
    in_model = sys.argv[1]
    img = sys.argv[2]
    preprocess = sys.argv[3] in accepted
    use_cuda = sys.argv[4] in accepted
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'

    if preprocess:
        ratio_width = 5
        ratio_height = 2
    else:
        ratio_width = 4
        ratio_height = 4

    model = LempiraNet(ratio_width, ratio_height)
    model.load_state_dict(torch.load(
        in_model, map_location=torch.device(device)))
    if use_cuda:
        model = model.to(device)
    model.eval()
    classes = [
        '1-frontal',
        '1-reverso',
        '2-frontal',
        '2-reverso',
        '5-frontal',
        '5-reverso',
        '10-frontal',
        '10-reverso',
        '20-frontal',
        '20-reverso',
        '50-frontal',
        '50-reverso',
        '100-frontal',
        '100-reverso',
        '200-frontal',
        '200-reverso',
        '500-frontal',
        '500-reverso'
    ]

    pil_img = Image.open(img)
    pil_img = safe_resize(pil_img, (64*ratio_width, 64*ratio_height))
    pil_img = TF.to_tensor(pil_img)
    pil_img = torch.unsqueeze(pil_img, 0)
    if use_cuda:
        pil_img.cuda()

    predicted_class = model(pil_img).detach().numpy()
    full_predict = classes[np.argmax(predicted_class)]
    print(full_predict)


if __name__ == '__main__':
    main()
