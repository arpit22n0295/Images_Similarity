from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tensorflow.keras.applications import VGG16
from PIL import Image 
import pickle 
import os

app = Flask(__name__)

def compute_similarity(image1, image2):
    similarity = cosine_similarity(image1.reshape(1, -1), image2.reshape(1, -1))[0][0]
    return similarity

# with open('load_function.pkl', 'rb') as f:
#     function= pickle.load(f)

def load_and_preprocess_image(Image_path):
    img = Image.open(Image_path)  # Load image
    img = img.resize((224, 224))  # Resize to match VGG-16 input
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)
    return img

# Load pre-trained VGG16 model
new_model = VGG16(weights=None, include_top=False,input_shape=(224, 224, 3)) 
new_model.load_weights('vgg_16.h5')

@app.route('/', methods=['GET'])
def welcome():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():

    if 'Image1' not in request.files or 'Image2' not in request.files:
        return render_template('index.html', error="Please select both images.")

    Image1 = request.files['Image1']
    Image2 = request.files['Image2']
    valid_extensions = ['.jpg', '.png', '.gif']
    

    Image1_path = os.path.join("E:/INTERN/Flask/Flask-Demo/", Image1.filename)
    Image1.save(Image1_path)
    Image2_path = os.path.join("E:/INTERN/Flask/Flask-Demo/", Image2.filename)
    Image1.save(Image2_path)


    Image1 = load_and_preprocess_image(Image1_path)
    Image2 = load_and_preprocess_image(Image2_path)

    embedding1 = new_model.predict([Image1])
    embedding2 = new_model.predict([Image2])

    similarity_score = compute_similarity(embedding1, embedding2)

    return str(similarity_score)

@app.route('/index', methods=['GET'])
def index():
    return "<H1> This is the Index Page</H2>"

if __name__ == '__main__':
    app.run(debug=True)
