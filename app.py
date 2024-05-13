from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from preprocessing import preprocess

app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template("index.html")


@app.post("/")
def enhance():
    file = request.files['vid']
    vid_file = f"./media/{secure_filename(file.filename)}"
    file.save(vid_file)
    enhanced_video = preprocess(vid_file)
    return render_template("result.html", enhanced_video)
