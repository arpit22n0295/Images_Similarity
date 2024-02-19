from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import VGG16
from PIL import Image 
import numpy as np

import pickle 

new_model = VGG16(weights='imagenet', include_top=False)

def load_and_preprocess_image(Image_path):
    img = Image.open(Image_path)  # Load image
    img = img.resize((224, 224))  # Resize to match VGG-16 input
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values
    print(img.shape)
    img = np.expand_dims(img, axis=0)
    return img


# with open('load_function.pkl', 'rb') as f:
#     function =pickle.load(f)

image=load_and_preprocess_image("E:/INTERN/Flask/Flask-Demo/wallpaper2.jpg")
print(image.shape)
embedding = new_model.predict([image])  # Get embedding from output of last convolutional layer
print(embedding.flatten().shape)

# C:\Users\aryaa\OneDrive\Desktop\Wallpapers\wallpaper2.jpg