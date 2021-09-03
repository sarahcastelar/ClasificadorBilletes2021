import os
import sys
import time
import math

import cv2
import numpy as np

from sklearn.decomposition import PCA

def removing_background(image, margen, iteraciones):
    # Basado en:
    # https://docs.opencv.org/master/d8/d83/tutorial_py_grabcut.html
    
    # original era hard coded, ahora se basa en margen
    # rectangle = (10, 10, 500, 225)
    rectangle = (margen, margen,
                 image.shape[1] - margen * 2,
                 image.shape[0] - margen * 2)
    
    # crear mascara y otros arreglos de salida ...
    mask = np.zeros(image.shape[:2], np.uint8)
    # 1x65 como lo requiere el algoritmo
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(image, mask, rectangle, bgdModel,
               fgdModel, iteraciones, cv2.GC_INIT_WITH_RECT)

    bin_bg_mask = (mask == 2) | (mask == 0)
    
    mask_2 = np.where(bin_bg_mask, 0, 1).astype('uint8')
    image = image * mask_2[:, :, np.newaxis]
    # cv.imshow("removing bg 1", image)
    
    return image, bin_bg_mask

def resizing_image(image, scale):
    # resizing the image
    width = int(round(image.shape[1] * scale))
    height = int(round(image.shape[0] * scale))

    dim = (width, height)
    
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # cv.imshow("Resized", image)
    
    return image


def process_file(input_filename):
    img = cv2.imread(input_filename)

    # si NO es horizontal ... rotar ... 
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Parametros
    margen = 10
    scale = 0.5
    iteraciones = 5

    # imagen_peque = img
    
    # Trabajar con una version mas pequenia de la imagen
    imagen_peque = resizing_image(img, scale)
    # cv.imshow("Resized", imagen_peque)
    
    # Aplicar segmentacion ....
    imagen_peque, bg_mask = removing_background(imagen_peque, margen,
                                                iteraciones)
    # cv2.imshow("Sin background", imagen_peque)
    # cv2.imshow("mascara", bg_mask.astype(np.uint8) * 255)
    # cv2.waitKey()

    # Obtener el area del billete
    fg_mask = np.logical_not(bg_mask).astype(np.uint8) * 255
    small_x, small_y, small_w, small_h = cv2.boundingRect(fg_mask)

    # small_cut = imagen_peque[small_y:small_y + small_h, small_x:small_x + small_w]
    # cv2.imshow("Small cut", small_cut)
    # cv2.waitKey()

    up_scale = 1.0 / scale 
    o_x = int(round(small_x * up_scale))
    o_y = int(round(small_y * up_scale))
    o_w = int(round(small_w * up_scale))
    o_h = int(round(small_h * up_scale))

    large_cut = img[o_y:o_y+o_h,o_x:o_x+o_w]
    # cv2.imshow("Large Cut", large_cut)
    # cv2.waitKey()

    # aplicar la mascara de fondo a la resolucion original
    o_dim = img.shape[1], img.shape[0]
    o_bg_mask = cv2.resize(bg_mask.astype(np.uint8) * 255, o_dim, interpolation=cv2.INTER_NEAREST)

    img_limpia = img.copy()
    img_limpia[o_bg_mask > 0, :] = 0, 0, 0

    limpia_cut = img_limpia[o_y:o_y+o_h,o_x:o_x+o_w]

    # encontrar el cuadrilatero que cubre todo
    o_fg_mask = 255 - o_bg_mask
    pixels_y, pixels_x = np.nonzero(o_fg_mask)
    pixels = np.vstack((pixels_x, pixels_y)).transpose()
    rr_center, rr_size, rr_angle = cv2.minAreaRect(pixels.astype(np.float32))

    if 5 <= rr_angle <= 85:
        print("Debe rotarse!")
        print((rr_angle, rr_size))

        cX = int(pixels[:,0].mean())
        cY = int(pixels[:,1].mean())
        r_h, r_w = img_limpia.shape[:2]

        if rr_size[0] > rr_size[1]:
            angulo = rr_angle
        else:
            angulo = -(90 - rr_angle)

        # reemplazar limpia por version rotada
        M = cv2.getRotationMatrix2D((cX, cY), angulo, 1.0)
        img_limpia = cv2.warpAffine(img_limpia, M, (r_w, r_h))

        # rotar mascara tambien...
        o_fg_mask = cv2.warpAffine(o_fg_mask, M, (r_w, r_h))

        # re-calcular los cortes
        o_x, o_y, o_w, o_h = cv2.boundingRect(o_fg_mask)

        large_cut = img[o_y:o_y+o_h,o_x:o_x+o_w]
        limpia_cut = img_limpia[o_y:o_y+o_h,o_x:o_x+o_w]

        # cv2.imshow("Antes", img_limpia)
        # cv2.imshow("Despues", rotated)
        # cv2.waitKey()
        

    """
    # Probando PCA para determinar el eje de mayor varianza
    # y si no esta cerca del eje horizontal... rotar!
    o_fg_mask = 255 - o_bg_mask
    pixels_y, pixels_x = np.nonzero(o_fg_mask)
    pixels = np.vstack((pixels_x, pixels_y)).transpose()

    pca = PCA(n_components=2)
    pca.fit(pixels)
    
    angulo_o = math.acos(pca.components_[0][0] * 1.0)
    angulo = angulo_o
    # poner en rango adecuado....
    while angulo < 0:
        angulo += math.pi
    while angulo > math.pi:
        angulo -= math.pi

    angulo_grados = (angulo / math.pi) * 180.0
    print(angulo_grados)
    if 10 <= angulo_grados <= 170:
        print("Debe rotarse!")
        print(angulo)
        print(pca.components_)

        cX = int(pca.mean_[0])
        cY = int(pca.mean_[1])
        r_h, r_w = img_limpia.shape[:2]

        angulo_o_grados = (angulo_o / math.pi) * 180.0
        print(angulo_o_grados)

        M = cv2.getRotationMatrix2D((cX, cY), -angulo_o_grados, 1.0)
        rotated = cv2.warpAffine(img_limpia, M, (r_w, r_h))

        cv2.imshow("Antes", img_limpia)
        cv2.imshow("Despues", rotated)
        cv2.waitKey()
        x = 0 / 0
    """

    # cv2.imshow("Large BG rem", img_limpia)
    # cv2.imshow("Large BG", o_bg_mask)
    # cv2.waitKey()
    

    """
    # recortando la imagen para que solo me salgan los colores principales, no el negro
    imagen_peque = cutting_black_bg(imagen_peque)
    """

    return large_cut, img_limpia, limpia_cut, input_filename

def main():
    if len(sys.argv) < 3:
        print("Uso:")
        print("\tpython {0:s} in_dir out_dir".format(sys.argv[0]))
        return

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    all_paths = [in_dir + "/" + name for name in os.listdir(in_dir)]

    start_time = time.time()

    for recorte_w_bg, full_no_bg, recorte_no_bg, in_path in map(process_file, all_paths[110:]):
        root_path, filename = os.path.split(in_path)

        out_path = out_dir + "/" + filename
        print(out_path)
        cv2.imwrite(out_path, recorte_no_bg)

    end_time = time.time()
    print("Tiempo: ", (end_time - start_time), " segundos")
    

if __name__ == "__main__":
    main()

