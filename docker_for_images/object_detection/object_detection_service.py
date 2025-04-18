import os
import io
import zipfile
import tempfile
import logging
import traceback
import shutil
import yaml
import uuid
import datetime
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Cache directory for model outputs
CACHE_DIR = os.path.join(tempfile.gettempdir(), "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Track temp files for cleanup
temp_dirs = []

# Cache metadata
cached_models = {}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "object-detection-service"})

@app.route('/finetune', methods=['POST'])
def finetune_model():
    try:
        if 'zipFile' not in request.files:
            return jsonify({"error": "No ZIP file uploaded"}), 400
        zip_file = request.files['zipFile']
        level = int(request.form.get('level', 3))
        manual_num = request.form.get('num_classes')
        training_id = str(uuid.uuid4())

        # Preset configs
        cfg = {
            1: {'model':'yolov8n.pt','epochs':5,'batch':16,'imgsz':160},
            2: {'model':'yolov8n.pt','epochs':10,'batch':16,'imgsz':320},
            3: {'model':'yolov8s.pt','epochs':30,'batch':16,'imgsz':640},
            4: {'model':'yolov8m.pt','epochs':80,'batch':8, 'imgsz':640},
            5: {'model':'yolov8l.pt','epochs':100,'batch':8,'imgsz':640}
        }
        if level not in cfg:
            return jsonify({"error":"Level must be between 1 and 5"}), 400
        params = cfg[level]

        # Unzip dataset
        temp_dir = tempfile.mkdtemp()
        temp_dirs.append(temp_dir)
        ds_dir = os.path.join(temp_dir, 'dataset')
        os.makedirs(ds_dir, exist_ok=True)
        zip_path = os.path.join(temp_dir, 'data.zip')
        zip_file.save(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(ds_dir)

        # Detect root folder
        subs = [d for d in os.listdir(ds_dir) if os.path.isdir(os.path.join(ds_dir, d))]
        root = os.path.join(ds_dir, subs[0]) if len(subs) == 1 else ds_dir

        # Build data.yaml
        class_ids = set()
        for split in ['train', 'valid']:
            lbl_dir = os.path.join(root, split, 'labels')
            for f in os.listdir(lbl_dir):
                if f.endswith('.txt'):
                    with open(os.path.join(lbl_dir, f), 'r') as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if parts:
                                class_ids.add(int(parts[0]))
        if not class_ids:
            raise ValueError("No class IDs found in labels")
        num_cls = int(manual_num) if manual_num else max(class_ids) + 1
        names = [f'class{i}' for i in range(num_cls)]

        data_yaml = {
            'path': root,
            'train': 'train/images',
            'val': 'valid/images',
            'nc': num_cls,
            'names': names
        }
        with open(os.path.join(root, 'data.yaml'), 'w') as f:
            yaml.dump(data_yaml, f)

        # Train with YOLOv8 API
        model = YOLO(params['model'])
        model.train(
            data=os.path.join(root, 'data.yaml'),
            epochs=params['epochs'],
            batch=params['batch'],
            imgsz=params['imgsz'],
            project=os.path.join(temp_dir, 'results'),
            name='exp',
            device='cpu'
        )

        # Locate best.pt
        out_dir = os.path.join(temp_dir, 'results', 'exp', 'weights')
        best = os.path.join(out_dir, 'best.pt')
        if not os.path.exists(best):
            raise FileNotFoundError(f"best.pt not found in {out_dir}")

        # Zip outputs
        out_zip = os.path.join(temp_dir, 'model.zip')
        with zipfile.ZipFile(out_zip, 'w') as z:
            z.write(best, 'best.pt')

        # Cache
        cache_path = os.path.join(CACHE_DIR, f"{training_id}.zip")
        shutil.copy(out_zip, cache_path)
        cached_models[training_id] = {
            'path': cache_path,
            'expires': datetime.datetime.now() + datetime.timedelta(hours=24)
        }

        # Send response
        buf = io.BytesIO(open(out_zip, 'rb').read())
        buf.seek(0)
        cleanup_temp_dirs()
        resp = send_file(buf, mimetype='application/zip', as_attachment=True, download_name='model.zip')
        resp.headers['X-Training-ID'] = training_id
        return resp

    except Exception as e:
        logger.error(traceback.format_exc())
        cleanup_temp_dirs()
        return jsonify({"error": str(e)}), 500

@app.route('/retrieve_model/<training_id>', methods=['GET'])
def retrieve_model(training_id):
    info = cached_models.get(training_id)
    if not info:
        return jsonify({"error": "Model not found or expired"}), 404
    if datetime.datetime.now() > info['expires']:
        os.remove(info['path'])
        del cached_models[training_id]
        return jsonify({"error": "Model expired and removed"}), 404
    return send_file(
        info['path'],
        mimetype='application/zip',
        as_attachment=True,
        download_name='model.zip'
    )


def cleanup_temp_dirs():
    global temp_dirs
    for d in temp_dirs:
        shutil.rmtree(d, ignore_errors=True)
    temp_dirs = []

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5027)))