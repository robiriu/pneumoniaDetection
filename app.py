from flask import Flask, request, render_template
import os
import tensorflow as tf
from utils.preprocess import preprocess_image  # Import preprocessing function

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = tf.keras.models.load_model('pneumonia_model.h5')

# Predict function
def predict_pneumonia(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    return "Pneumonia Detected" if prediction[0][0] > 0.5 else "Normal"

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            result = predict_pneumonia(filepath)
            return render_template('index.html', result=result, image=filepath)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
