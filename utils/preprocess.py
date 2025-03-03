from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img_path, target_size=(224, 224)):
    """
    Preprocess an image for model prediction.
    Args:
        img_path (str): Path to the input image.
        target_size (tuple): Target size for resizing (default is 224x224).
    Returns:
        numpy.ndarray: Preprocessed image ready for model input.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
