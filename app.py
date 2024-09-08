import os
import zipfile
import tempfile
import threading
import uuid
from flask import Flask, render_template, request, jsonify
import fal_client

app = Flask(__name__)

# Use environment variable for FAL_KEY
fal_client.api_key = os.environ.get('FAL_KEY')

# Temporary storage for uploaded files
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dictionary to store job statuses
job_statuses = {}

def run_training_job(job_id, images_url):
    try:
        handler = fal_client.submit(
            "fal-ai/flux-lora-general-training",
            arguments={
                "images_data_url": images_url
            },
        )
        result = handler.get()
        model_url = result['diffusers_lora_file']['url']
        job_statuses[job_id] = {'status': 'completed', 'model_url': model_url}
    except Exception as e:
        job_statuses[job_id] = {'status': 'failed', 'error': str(e)}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        return jsonify({'message': 'File uploaded successfully', 'filename': file.filename})

@app.route('/train', methods=['POST'])
def train():
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400

        # Create a zip file containing all uploaded images
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], 'images.zip')
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for file in files:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                zip_file.write(file_path, file)

        # Upload the zip file to fal storage
        with open(zip_path, 'rb') as f:
            url = fal_client.upload(f, "application/zip")

        job_id = str(uuid.uuid4())
        job_statuses[job_id] = {'status': 'running'}
        
        # Start the training job in a background thread
        thread = threading.Thread(target=run_training_job, args=(job_id, url))
        thread.start()

        # Clean up: remove all uploaded files and the zip file
        for file in files:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
        os.remove(zip_path)

        return jsonify({'job_id': job_id, 'status': 'training_started'})
    except Exception as e:
        app.logger.error(f"An error occurred during training: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/job_status/<job_id>', methods=['GET'])
def job_status(job_id):
    status = job_statuses.get(job_id, {'status': 'not_found'})
    return jsonify(status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))