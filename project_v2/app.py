from flask import Flask, request, render_template, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
from modules.sam_handler import run_sam_on_image
from modules.gpt_handler import gpt_analyze_image

app = Flask(__name__)
app.secret_key = 'your-secret-key'

UPLOAD_FOLDER = 'static/uploaded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'Empty filename', 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    action = request.form.get('action')
    api_key = request.form.get('api_key')

    if action == 'gpt':
        result = gpt_analyze_image(image_path, api_key)
        return render_template('result.html', result_type='GPT', image=filename, result=result)

    elif action == 'sam':
        result = run_sam_on_image(image_path)
        return render_template('result.html', result_type='SAM', image=os.path.basename(result['visualization']), result=result['regions'])

    else:
        return 'Invalid action', 400

if __name__ == '__main__':
    app.run(debug=True)
