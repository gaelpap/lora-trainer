import threading
from flask import Flask, jsonify, request
import fal_client
import uuid

app = Flask(__name__)

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

@app.route('/train', methods=['POST'])
def train():
    # Your existing code to create and upload zip file
    # ...

    job_id = str(uuid.uuid4())
    job_statuses[job_id] = {'status': 'running'}
    
    # Start the training job in a background thread
    thread = threading.Thread(target=run_training_job, args=(job_id, url))
    thread.start()

    return jsonify({'job_id': job_id, 'status': 'training_started'})

@app.route('/job_status/<job_id>', methods=['GET'])
def job_status(job_id):
    status = job_statuses.get(job_id, {'status': 'not_found'})
    return jsonify(status)