# pipeline.py

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import re
import mediapipe as mp
import math
from ultralytics import YOLO

COIN_DIAMETERS = {
    "10sen-1stseries": 19.00, "10sen-2ndseries": 19.40, "10sen-3rdseries": 18.80,
    "20sen-1stseries": 23.00, "20sen-2ndseries": 23.59, "20sen-3rdseries": 20.60,
    "50sen-1stseries": 28.00, "50sen-2ndseries": 27.76, "50sen-3rdseries": 22.65,
    "5sen-1stseries": 16.00,  "5sen-2ndseries": 16.25,  "5sen-3rdseries": 17.78
}

class CoinMeasurementPipeline:
    def __init__(self, detection_model_path, classification_model_path, labels_path):
        # ... (keep all methods exactly as they were: __init__, load_detection_model, etc.)
        """Initializes the pipeline by loading necessary models."""
        self.coin_diameters = COIN_DIAMETERS
        self.detection_model = None
        self.classification_model = None
        self.labels = []
        
        self.load_detection_model(detection_model_path)
        self.load_classification_model(classification_model_path, labels_path)

    def load_detection_model(self, model_path):
        """Loads the YOLO object detection model."""
        try:
            self.detection_model = YOLO(model_path)
            print("Object detection model (YOLO) loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load detection model: {e}")

    def load_classification_model(self, model_path, labels_path):
        """Loads the TensorFlow classification model and labels."""
        try:
            self.classification_model = tf.keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            print("TensorFlow classification model and labels loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load classification model: {e}")

    def detect_coins(self, image):
        # ... (no changes needed)
        """Detects coins in the image and returns their bounding boxes."""
        if self.detection_model is None:
            print("   ERROR: Detection model is not loaded.")
            return None
        try:
            results = self.detection_model(image, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf > 0.5:  # Confidence threshold
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cls = int(box.cls[0])
                        class_name = self.detection_model.names[cls]
                        detections.append({'bbox': (x1, y1, x2, y2), 'confidence': conf, 'class_name': class_name})
            
            print(f"   -> Found {len(detections)} potential coin(s).")
            return detections
        except Exception as e:
            print(f"   ERROR: An exception occurred during coin detection: {e}")
            return None
    
    # ... include ALL other methods from your class here without changes ...
    # (extract_coin_image, classify_coin, get_coin_diameter, 
    #  calculate_coin_pixel_diameter, analyze_head_and_estimate_circumference)
    def extract_coin_image(self, original_image, bbox):
        x1, y1, x2, y2 = bbox
        return original_image[y1:y2, x1:x2]

    def classify_coin(self, coin_image):
        if self.classification_model is None: return None, 0.0
        try:
            coin_pil = Image.fromarray(cv2.cvtColor(coin_image, cv2.COLOR_BGR2RGB)).resize((224, 224))
            img_array = np.array(coin_pil)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = (img_array.astype(np.float32) / 127.5) - 1
            predictions = self.classification_model(img_array, training=False)
            if isinstance(predictions, dict): predictions = list(predictions.values())[0]
            predictions = predictions.numpy()
            idx = np.argmax(predictions[0])
            confidence = float(predictions[0][idx])
            raw_class = self.labels[idx].strip()
            cleaned_class = re.sub(r'^\s*\d+\s*', '', raw_class).replace('-front', '').replace('-back', '').replace('-end', '').strip()
            return cleaned_class, confidence
        except Exception as e: return None, 0.0

    def get_coin_diameter(self, coin_class):
        return self.coin_diameters.get(coin_class, None)

    def calculate_coin_pixel_diameter(self, coin_image):
        if coin_image is None or coin_image.size == 0: return 0
        h, w = coin_image.shape[:2]
        gray = cv2.cvtColor(coin_image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(denoised, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(h*0.8), param1=50, param2=30, minRadius=int(w/6), maxRadius=int(w/2 + 5))
        if circles is not None:
            return int(np.round(circles[0, 0, 2]).astype("int") * 2)
        _, th = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            (_, _), radius = cv2.minEnclosingCircle(cnt)
            return int(radius * 2)
        return 0

    def analyze_head_and_estimate_circumference(self, image, mm_per_pixel):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(img_rgb)
        if not results or not results.multi_face_landmarks: return None
        face_landmarks = results.multi_face_landmarks[0]
        landmarks_points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
        seed_y = landmarks_points[10][1]
        dynamic_roi_offset = int(h * (200 / 2560.0))
        roi = image[max(0, seed_y - dynamic_roi_offset):min(h, seed_y + dynamic_roi_offset), :]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(roi_gray, 50, 150)
        kernel = np.ones((5,5), np.uint8)
        closed_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            candidates = [c for c in contours if cv2.contourArea(c) > 100 and (cv2.boundingRect(c)[2] / float(cv2.boundingRect(c)[3]) > 3 if cv2.boundingRect(c)[3] > 0 else False)]
            if candidates:
                band_contour = max(candidates, key=lambda c: cv2.arcLength(c, True))
                mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [band_contour], -1, 255, -1)
                try:
                    skeleton = cv2.ximgproc.thinning(mask)
                    arc_length_px = float(cv2.countNonZero(skeleton))
                except Exception:
                    arc_length_px = float(cv2.arcLength(band_contour, True))
                leftmost = tuple(band_contour[band_contour[:, :, 0].argmin()][0])
                rightmost = tuple(band_contour[band_contour[:, :, 0].argmax()][0])
                chord_length_px = math.hypot(rightmost[0] - leftmost[0], rightmost[1] - leftmost[1])
                if chord_length_px > 0:
                    p_left_temple, p_right_temple = landmarks_points[454], landmarks_points[234]
                    full_head_width_px = math.hypot(p_right_temple[0] - p_left_temple[0], p_right_temple[1] - p_left_temple[1])
                    intricacy_index = arc_length_px / chord_length_px
                    estimated_full_frontal_arc_px = full_head_width_px * intricacy_index
                    estimated_circumference_px = estimated_full_frontal_arc_px * 2.0
                    final_circumference_cm = (estimated_circumference_px * mm_per_pixel) / 10.0
                    return {"head_circumference_cm": final_circumference_cm}
        return None

    # *** CRITICAL CHANGE IS HERE ***
    def process_image(self, image_cv): # Changed argument from image_path to image_cv
        """Main orchestrator for the entire measurement pipeline."""
        status = {'pipelineSuccess': False, 'errorMessage': "", 'finalResults': {}}
        
        # REMOVED: image = cv2.imread(image_path)
        # The image is now passed directly to the function.
        
        detections = self.detect_coins(image_cv)
        if not detections:
            status['errorMessage'] = "No coins were detected in the image."
            return status
        
        detection = max(detections, key=lambda d: d['confidence'])
        coin_image = self.extract_coin_image(image_cv, detection['bbox'])
        coin_class, confidence = self.classify_coin(coin_image)
        if coin_class is None:
            status['errorMessage'] = "Coin classification failed."
            return status

        actual_diameter_mm = self.get_coin_diameter(coin_class)
        if actual_diameter_mm is None:
            status['errorMessage'] = f"Unknown coin class '{coin_class}' for calibration."
            return status

        pixel_diameter = self.calculate_coin_pixel_diameter(coin_image)
        if pixel_diameter <= 0:
            status['errorMessage'] = "Failed to calculate the coin's pixel diameter."
            return status

        mm_per_pixel = actual_diameter_mm / pixel_diameter
        final_result = self.analyze_head_and_estimate_circumference(image_cv, mm_per_pixel)
        
        if final_result is None:
            status['errorMessage'] = "Failed to estimate head circumference. The rubber band may not be clearly visible. It is advised to use a rubber band that contrasts with your skin/hair colour (e.g., green)."
            return status

        status['pipelineSuccess'] = True
        status['finalResults'] = {'coin_class': coin_class, 'confidence': confidence, 'mm_per_pixel': mm_per_pixel, **final_result}
        
        circumference = final_result.get('head_circumference_cm')
        if circumference is not None and circumference < 20:
            status['qualityWarning'] = ("Measurement may be inaccurate due to small calculated value. Please retake the picture, ensuring the rubber band is clearly visible across the forehead.")
        
        return status