import numpy as np
from sklearn.cluster import KMeans

def getting_color_values_v1(image):
    # hacer clustering de colores
    colores = list()
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=7)
    clt.fit(image)

    # saca todos los colores y hace un histograma ...
    # basicamente, es la cuenta de pixeles por cluster de color
    # .reshape(1, -1)
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    #bar = np.zeros((50, 300, 3), dtype="uint8")
    #startX = 0

    # aqui se sacan simplemente los colores (clusters)
    # pero el histograma NO se usa
    for (percent, color) in zip(hist, clt.cluster_centers_):  # BGR
        b = round(((color[0] / 255) - 0.406) / 0.225,3)
        g = round(((color[1] / 255) - 0.456) / 0.224,3)
        r = round(((color[2] / 255) - 0.485) / 0.229,3)
        colores.append(b)
        colores.append(g)
        colores.append(r)
        #endX = startX + (percent * 300)
        #cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
        #             color.astype("uint8").tolist(), -1)
        #startX = endX
    # # print(final_colors)
    # cv.imshow("palette", bar)
    # cv.waitKey(0)

    return colores

def getting_color_values_v2(image):
    # hacer clustering de colores
    colores = list()
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=7)
    clt.fit(image)

    # contar ocurrencia de cada color
    labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    colors = []
    for porcentaje,color in zip(hist, clt.cluster_centers_):
        colors.append((porcentaje,color))
    colors = sorted(colors,key=lambda x: x[0], reverse=True)

    features = []
    for (porcentaje, color) in colors:
        #Normalizando colores
        features.append(round(((color[0] / 255) - 0.406) / 0.225,3))
        features.append(round(((color[1] / 255) - 0.456) / 0.224,3))
        features.append(round(((color[2] / 255) - 0.485) / 0.229,3))

    return features


