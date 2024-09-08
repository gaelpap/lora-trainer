import os
import zipfile
import tempfile
import threading
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
import fal_client

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_secret_key_here')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///site.db')
if app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

fal_client.api_key = os.environ.get('FAL_KEY')

UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    jobs = db.relationship('Job', backref='user', lazy=True)

class Job(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    status = db.Column(db.String(20), nullable=False, default='running')
    model_url = db.Column(db.String(200))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def run_training_job(job_id, images_url, user_id):
    try:
        handler = fal_client.submit(
            "fal-ai/flux-lora-general-training",
            arguments={
                "images_data_url": images_url
            },
        )
        result = handler.get()
        model_url = result['diffusers_lora_file']['url']
        job = Job.query.get(job_id)
        job.status = 'completed'
        job.model_url = model_url
        db.session.commit()
    except Exception as e:
        job = Job.query.get(job_id)
        job.status = 'failed'
        db.session.commit()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already registered.')
            return redirect(url_for('register'))
        new_user = User(email=email, password=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    jobs = Job.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', jobs=jobs)

@app.route('/upload', methods=['POST'])
@login_required
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
@login_required
def train():
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400

        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], 'images.zip')
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for file in files:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                zip_file.write(file_path, file)

        with open(zip_path, 'rb') as f:
            url = fal_client.upload(f, "application/zip")

        job_id = str(uuid.uuid4())
        new_job = Job(id=job_id, user_id=current_user.id)
        db.session.add(new_job)
        db.session.commit()
        
        thread = threading.Thread(target=run_training_job, args=(job_id, url, current_user.id))
        thread.start()

        for file in files:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
        os.remove(zip_path)

        return jsonify({'job_id': job_id, 'status': 'training_started'})
    except Exception as e:
        app.logger.error(f"An error occurred during training: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/job_status/<job_id>', methods=['GET'])
@login_required
def job_status(job_id):
    job = Job.query.get(job_id)
    if job and job.user_id == current_user.id:
        return jsonify({'status': job.status, 'model_url': job.model_url})
    return jsonify({'status': 'not_found'}), 404

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        email = request.form.get('email')
        new_password = request.form.get('new_password')
        user = User.query.filter_by(email=email).first()
        if user:
            user.password = generate_password_hash(new_password)
            db.session.commit()
            flash('Password has been reset. Please login with your new password.')
            return redirect(url_for('login'))
        else:
            flash('Email not found.')
    return render_template('reset_password.html')

with app.app_context():
    db.create_all()