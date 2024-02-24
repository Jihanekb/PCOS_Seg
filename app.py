import os
from flask import Flask, request, jsonify, send_file
from PIL import Image
from ultralytics import YOLO
import werkzeug
import numpy as np
import io


app = Flask(__name__)

# Load the trained model
model = YOLO("model/best.pt")
confidence = 0.4


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            # Read image from memory
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))

            # Perform object detection
            try:
                results = model(img, conf=confidence, save=False)
                for i, r in enumerate(results):
                    detected_img = r.plot(pil=True)  # # Plot results image
                    detected_pil_img = Image.fromarray(detected_img)
            except Exception as ex:
                return jsonify({'error': f'Error during segmentation: {ex}'})
            

            # Convert image to bytes
            img_byte_array = io.BytesIO()
            detected_pil_img.save(img_byte_array, format='PNG')
            img_byte_array = img_byte_array.getvalue()

            return send_file(io.BytesIO(img_byte_array), mimetype='image/png')

            
    
    return """
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method="post" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)