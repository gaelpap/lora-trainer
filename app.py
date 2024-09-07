import os
import zipfile
import tempfile
from flask import Flask, render_template, request, jsonify
import fal_client

app = Flask(__name__)

# Use environment variable for FAL_KEY
fal_client.api_key = os.environ.get('FAL_KEY')

# Temporary storage for uploaded files
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            return jsonify({'message': 'File uploaded successfully', 'filename': file.filename})
    
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    if not files:
        return jsonify({'error': 'No files uploaded'})

    # Create a zip file containing all uploaded images
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], 'images.zip')
    with zipfile.ZipFile(zip_path, 'w') as zip_file:
        for file in files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
            zip_file.write(file_path, file)

    # Upload the zip file to fal storage
    with open(zip_path, 'rb') as f:
        url = fal_client.upload(f, "application/zip")

    # Call the API
    try:
        handler = fal_client.submit(
            "fal-ai/flux-lora-general-training",
            arguments={
                "images_data_url": url
            },
        )

        result = handler.get()

        # Extract the .safetensors file URL
        model_url = result['diffusers_lora_file']['url']

        # Clean up: remove all uploaded files and the zip file
        for file in files:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
        os.remove(zip_path)

        return jsonify({'model_url': model_url})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))