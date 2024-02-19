from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET'])
def upload_form():
    return render_template('p.html')


@app.route('/upload', methods=['POST'])
def process_image():
    # Get uploaded image file
    image_file = request.files['image']

    # Save the image file (adapt path and naming as needed)
    image_path = f"E:/INTERN/Flask/Flask-Demo/templates/{image_file.filename}"
    image_file.save(image_path)

    # Now you can use the image_path in your Python code to process it using VGG16 or other methods

    return f"Image uploaded successfully! Path: {image_path}"

if __name__ == '__main__':
    app.run(debug=True)
