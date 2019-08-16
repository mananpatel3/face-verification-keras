import flask_compare as cmp
from scipy.spatial import distance
from PIL import Image
import numpy as np

cmp.init("facenet_keras.h5")


def single_image(img_1, img_2):
    #print(img_1)
    #print(img_2)
    img_1 = crop_image(img_1)
    img_2 = crop_image(img_2)
    embd1 = cmp.calc_embs(img_1)
    embd2 = cmp.calc_embs(img_2)

    dist = calc_dist(embd1, embd2)
    #print(dist)
    same = ("Authentication Successful")
    different = ("Authentication Failed")
    if dist < 0.65:
        print(same)
        return same
    elif dist > 1.00:
        print(different)
        return different
    else:
        global count
        cosembd1 = cmp.calc_embs(img_1)
        cosembd2 = cmp.calc_embs(img_2)

        cos_similarity = calc_cosine(cosembd1, cosembd2)
        if (cos_similarity > 0.70):
            print(same)
            return same
        else:
            print(different)
            return different

def calc_dist(embd1, embd2):
    dist = distance.euclidean(embd1, embd2)
    print("Distance  : {}".format(dist))
    return dist

def crop_image(img):
    img = Image.fromarray(img)
    img = img.resize((160,160),Image.ANTIALIAS)
    return np.array(img)


def calc_cosine(source_representation, test_representation):
    result = 1 - distance.cosine(source_representation[0], test_representation[0])
    print("Cosine Similarity : {}".format(result))
    return result

# if __name__ == '__main__':
# single_image()
