from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from preprocessing import preprocess
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.post("/")
def enhance():
    file = request.files['vid']
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
    vid_file = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    enhanced_video = preprocess(vid_file)
    return render_template("result.html", enhanced_vid=enhanced_video)
