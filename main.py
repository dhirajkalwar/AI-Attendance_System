import os
import io
import re
import time
import gc
import threading
import concurrent.futures
import datetime

import cv2
import numpy as np
import torch
import torchvision.ops as ops
import tensorflow as tf
import tensorflow_hub as hub
import easyocr
import numpy as np
import datetime

# ---------------------------
# Google Drive API Imports (for script mode)
# ---------------------------
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
except ImportError:
    # Not needed in Colab mode if not running as a script.
    pass

# ---------------------------
# Colab-specific Imports (only available in Colab)
# ---------------------------
try:
    from google.colab import files
    from google.colab.patches import cv2_imshow
except ImportError:
    pass

# ---------------------------
# ReportLab for PDF Report
# ---------------------------
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image as RLImage
from reportlab.lib import colors

# ---------------------------
# Global Variables for Models and Concurrency Locks
# ---------------------------
sr = None
yolo_model = None
effdet_model = None

yolo_lock = threading.Lock()
effdet_lock = threading.Lock()
sr_lock = threading.Lock()

# ---------------------------
# Model Initialization Function
# ---------------------------
def init_models():
    global sr, yolo_model, effdet_model
    # Super Resolution Model (requires EDSR_x4.pb to be in the working directory)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("EDSR_x4.pb")
    sr.setModel("edsr", 4)  # Using a scale factor of 4

    # YOLO Model using ultralytics (make sure yolov8x.pt is available)
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Please install ultralytics (`pip install ultralytics`) to run YOLO detection.")
    yolo_model = YOLO('yolov8x.pt')

    # EfficientDet model from TensorFlow Hub
    effdet_model = hub.load('https://tfhub.dev/tensorflow/efficientdet/d7/1')
    print("Models have been initialized.")

# ---------------------------
# Google Drive Functions (for script mode)
# ---------------------------
def list_images(folder_id, drive_service):
    query = f"'{folder_id}' in parents and mimeType contains 'image/'"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files_list = results.get('files', [])
    return files_list

def download_image(file_id, file_name, drive_service):
    folder_path = "./images"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    request = drive_service.files().get_media(fileId=file_id)
    with io.FileIO(file_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    print(f"✅ Downloaded: {file_name} → {file_path}")
    return file_path

# ---------------------------
# OCR Function to Extract Lab Number
# ---------------------------
def process_ocr(image_path):
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"Could not load image: {image_path}")
        return "N/A"
    reader = easyocr.Reader(['en'], gpu=False)
    ocr_results = reader.readtext(image_cv, detail=0)
    combined_text = " ".join(ocr_results)
    # Look for three-digit numbers (adjust regex if needed)
    lab_numbers = re.findall(r'\b\d{3}\b', combined_text)
    if lab_numbers:
        # Use the last found lab number
        lab_number = lab_numbers[-1]
        return lab_number
    else:
        return "N/A"

# ---------------------------
# Detection Functions: YOLO & EfficientDet on Image Patches
# ---------------------------
def detect_yolo_multiscale(patch, yolo_model, scales=[0.8, 1.0, 1.2], conf_threshold=0.4):
    boxes = []
    scores = []
    for scale in scales:
        scaled_patch = cv2.resize(patch, None, fx=scale, fy=scale)
        with yolo_lock:
            results = yolo_model(scaled_patch, conf=conf_threshold)
        for box in results[0].boxes:
            cls = int(box.cls[0].item())
            if yolo_model.names[cls] == "person":
                x, y, w, h = box.xywh[0].cpu().numpy()
                x, y, w, h = x/scale, y/scale, w/scale, h/scale
                x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
                boxes.append([x1, y1, x2, y2])
                scores.append(box.conf[0].item())
    if boxes:
        return np.array(boxes), np.array(scores)
    else:
        return np.empty((0, 4)), np.empty((0,))

def detect_effdet_patch(patch, effdet_model, conf_threshold=0.4):
    patch_h, patch_w = patch.shape[:2]
    eff_image = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    input_image = tf.convert_to_tensor(eff_image)
    input_image = tf.expand_dims(input_image, axis=0)
    with effdet_lock:
        eff_results = effdet_model(input_image)
    boxes_tf = eff_results['detection_boxes'].numpy()[0]
    scores_tf = eff_results['detection_scores'].numpy()[0]
    classes_tf = eff_results['detection_classes'].numpy()[0].astype(np.int32)
    person_mask = np.logical_and(classes_tf == 1, scores_tf >= conf_threshold)
    eff_boxes_norm = boxes_tf[person_mask]
    eff_scores = scores_tf[person_mask]
    eff_boxes = []
    for box in eff_boxes_norm:
        ymin, xmin, ymax, xmax = box
        x1 = int(xmin * patch_w)
        y1 = int(ymin * patch_h)
        x2 = int(xmax * patch_w)
        y2 = int(ymax * patch_h)
        eff_boxes.append([x1, y1, x2, y2])
    if eff_boxes:
        return np.array(eff_boxes), np.array(eff_scores)
    else:
        return np.empty((0, 4)), np.empty((0,))

def detect_on_patch(patch, yolo_model, effdet_model, conf_threshold=0.2):
    yolo_boxes, yolo_scores = detect_yolo_multiscale(patch, yolo_model, conf_threshold=conf_threshold)
    eff_boxes, eff_scores = detect_effdet_patch(patch, effdet_model, conf_threshold=conf_threshold)

    if yolo_boxes.shape[0] == 0 and eff_boxes.shape[0] == 0:
        ensemble_boxes = np.empty((0, 4))
        ensemble_scores = np.empty((0,))
    else:
        if yolo_boxes.shape[0] > 0 and eff_boxes.shape[0] > 0:
            ensemble_boxes = np.vstack([yolo_boxes, eff_boxes])
            ensemble_scores = np.concatenate([yolo_scores, eff_scores])
        elif yolo_boxes.shape[0] > 0:
            ensemble_boxes = yolo_boxes
            ensemble_scores = yolo_scores
        else:
            ensemble_boxes = eff_boxes
            ensemble_scores = eff_scores

    if ensemble_boxes.shape[0] > 0:
        boxes_tensor = torch.tensor(ensemble_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(ensemble_scores, dtype=torch.float32)
        nms_indices = ops.nms(boxes_tensor, scores_tensor, 0.4)
        final_boxes = boxes_tensor[nms_indices].numpy()
        final_scores = scores_tensor[nms_indices].numpy()
    else:
        final_boxes = np.empty((0, 4))
        final_scores = np.empty((0,))
    return final_boxes, final_scores

# ---------------------------
# Full Image Processing
# ---------------------------
def process_file(fn):
    print(f"Processing file: {fn}")
    orig_image = cv2.imread(fn)
    if orig_image is None:
        print(f"Error reading {fn}")
        return fn, None, None, 0

    with sr_lock:
        enhanced_image = sr.upsample(orig_image)
    enh_h, enh_w = enhanced_image.shape[:2]
    mid_x, mid_y = enh_w // 2, enh_h // 2

    quadrants = {
        'top_left':    (enhanced_image[0:mid_y, 0:mid_x], 0, 0),
        'top_right':   (enhanced_image[0:mid_y, mid_x:enh_w], mid_x, 0),
        'bottom_left': (enhanced_image[mid_y:enh_h, 0:mid_x], 0, mid_y),
        'bottom_right':(enhanced_image[mid_y:enh_h, mid_x:enh_w], mid_x, mid_y)
    }

    total_count = 0
    all_boxes_global = []
    all_scores_global = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as quad_executor:
        future_to_quad = {}
        for quad_name, (quad_img, offset_x, offset_y) in quadrants.items():
            future = quad_executor.submit(detect_on_patch, quad_img, yolo_model, effdet_model, 0.2)
            future_to_quad[future] = (quad_name, offset_x, offset_y)
        for future in concurrent.futures.as_completed(future_to_quad):
            quad_name, offset_x, offset_y = future_to_quad[future]
            quad_boxes, quad_scores = future.result()
            count_quad = quad_boxes.shape[0]
            total_count += count_quad
            for box in quad_boxes:
                x1, y1, x2, y2 = box
                all_boxes_global.append([x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y])
            for score in quad_scores:
                all_scores_global.append(score)
            print(f"{quad_name}: Detected {count_quad} persons.")

    print(f"Total persons detected in {fn}: {total_count}")

    output_image = enhanced_image.copy()
    for i, box in enumerate(all_boxes_global):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person: {all_scores_global[i]:.2f}"
        cv2.putText(output_image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    del orig_image, enhanced_image, quadrants, all_boxes_global, all_scores_global
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fn, output_image, total_count, None

# ---------------------------
# PDF Report Generation Function
# ---------------------------
def create_pdf(data, filename="output.pdf", header_image=None):
    margin = 10
    document = SimpleDocTemplate(filename, pagesize=letter,
                                 rightMargin=margin, leftMargin=margin,
                                 topMargin=margin, bottomMargin=margin)
    elements = []

    # Add header image if exists
    if header_image:
        img = RLImage(header_image, width=letter[0]-2*margin, height=100)
        elements.append(img)

    # Add generated date
    generated_date = datetime.datetime.now().strftime("%B %d, %Y %I:%M %p")
    date_table = Table([[f"Generated on: {generated_date}"]],
                      colWidths=[document.pagesize[0]-2*margin])
    date_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'RIGHT'),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.grey),
        ('BOTTOMPADDING', (0,0), (-1,-1), 20),
    ]))
    elements.append(date_table)

    # Add main data table
    table_data = [["Lab No", "Head Count", "Time"]]
    table_data.extend(data)
    table = Table(table_data, colWidths=[document.pagesize[0]*0.3]*3)
    table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,0), (-1,0), (0.7, 0.7, 0.7)),
        ('GRID', (0,0), (-1,-1), 0.5, (0,0,0)),
        ('LINEBELOW', (0,0), (-1,0), 1, (0,0,0))
    ]))
    elements.append(table)
    document.build(elements)
    print(f"PDF report created: {filename}")

# ---------------------------
# Main Function for Colab Mode (File Upload)
# ---------------------------
def run_colab():
    print("Running in Colab mode. Please upload your images:")
    uploaded = files.upload()

    lab_results = {}  # key: lab number, value: (output_image, total_count, time, output_filename)
    init_models()

    for filename, file_content in uploaded.items():
        with open(filename, 'wb') as f:
            f.write(file_content)

        lab_no = process_ocr(filename)
        fn, output_image, total_count, _ = process_file(filename)
        output_filename = "processed_" + filename
        cv2.imwrite(output_filename, output_image)
        current_time = time.strftime("%I:%M %p")
        lab_results[lab_no] = (output_image, total_count, current_time, output_filename)

    results_data = []
    for lab_no, (output_image, total_count, current_time, output_filename) in lab_results.items():
        results_data.append([lab_no, total_count, current_time])

    header_img = "tcetLogo.jpg" if os.path.exists("tcetLogo.jpg") else None
    create_pdf(results_data, filename="output.pdf", header_image=header_img)

# ---------------------------
# Main Function for Script Mode (Google Drive Integration)
# ---------------------------
def run_script():
    SERVICE_ACCOUNT_FILE = 'credential.json'
    SCOPES = ['https://www.googleapis.com/auth/drive']
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=credentials)

    folder_id = "1kyPaOUn2Qr2ZAE0F-atr29FmzHUazk7d"
    files_list = list_images(folder_id, drive_service)
    if not files_list:
        print("No images found in the specified folder.")
        return

    init_models()
    lab_results = {}  # key: lab number, value: (output_image, total_count, time, output_filename)

    for file in files_list:
        file_id = file['id']
        file_name = file['name']
        local_path = download_image(file_id, file_name, drive_service)

        lab_no = process_ocr(local_path)
        fn, output_image, total_count, _ = process_file(local_path)
        output_filename = "processed_" + file_name
        cv2.imwrite(output_filename, output_image)
        current_time = time.strftime("%I:%M %p")
        lab_results[lab_no] = (output_image, total_count, current_time, output_filename)

    results_data = []
    for lab_no, (output_image, total_count, current_time, output_filename) in lab_results.items():
        results_data.append([lab_no, total_count, current_time])

    header_img = "tcetLogo.jpg" if os.path.exists("tcetLogo.jpg") else None
    create_pdf(results_data, filename="output.pdf", header_image=header_img)

# ---------------------------
# Entry Point: Choose Mode Based on Environment
# ---------------------------
if __name__ == "__main__":
    try:
        # If running in Colab, the google.colab module will be available
        import google.colab  # noqa
        run_colab()
        # run_script()  # Uncomment if you want to run script mode in Colab
    except ImportError:
        # Otherwise, assume running as a standalone script
        run_script()