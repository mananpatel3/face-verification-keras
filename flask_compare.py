import numpy as np
from keras.models import load_model
import tensorflow as tf
global graph, model
graph = tf.get_default_graph()
image_size = 160   #160

model = None

def init(facenet_model_path):
    global model
    model = load_model(facenet_model_path)

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def load_and_resize_image(filepath):

    #aligned_images = []
    #img = cv2.imread(filepath)
    #b, g, r = cv2.split(img)  # get b,g,r
    #rgb_img = cv2.merge([r, g, b])  # switch it to rgb
    #aligned = cv2.resize(rgb_img,(image_size, image_size))
    #aligned = load_img(filepath, target_size=(image_size))
    return np.array([filepath])

def calc_embs(filepath):
    aligned_images = prewhiten(load_and_resize_image(filepath))
    with graph.as_default():
        predected_val = model.predict(aligned_images)
        embs = l2_normalize(predected_val)
        return embs





