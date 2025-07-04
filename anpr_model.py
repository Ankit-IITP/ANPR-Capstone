# anpr_model.py

import os
import cv2
import numpy as np
import tensorflow as tf
import easyocr
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder

# Setup paths
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
WORKSPACE_PATH = os.path.join('Tensorflow', 'workspace')
CHECKPOINT_PATH = os.path.join(WORKSPACE_PATH, 'models', CUSTOM_MODEL_NAME)
PIPELINE_CONFIG = os.path.join(CHECKPOINT_PATH, 'pipeline.config')
LABELMAP_PATH = os.path.join(WORKSPACE_PATH, 'annotations', 'label_map.pbtxt')

# Load pipeline config and build model
configs = config_util.get_configs_from_pipeline_file(PIPELINE_CONFIG)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-4')).expect_partial()

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(LABELMAP_PATH)

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def detect_plate(image_path):
    image_np = cv2.imread(image_path)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    # Visualization
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        (detections['detection_classes'] + 1).numpy().astype(int),
        detections['detection_scores'].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=0.8,
        agnostic_mode=False
    )

    return image_np_with_detections
reader = easyocr.Reader(['en'])  # Load once

import easyocr
reader = easyocr.Reader(['en'])

def detect_plate(image_path):
    import cv2
    import numpy as np
    import tensorflow as tf

    image_np = cv2.imread(image_path)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()

    h, w, _ = image_np.shape

    # Step 1: Try using detection boxes
    for i in range(len(scores)):
        if scores[i] > 0.5:
            ymin, xmin, ymax, xmax = boxes[i]
            left, right = int(xmin * w), int(xmax * w)
            top, bottom = int(ymin * h), int(ymax * h)

            cropped = image_np[top:bottom, left:right]
            cv2.imwrite("static/uploads/debug_crop.jpg", cropped)

            results = reader.readtext(cropped)
            print("OCR on cropped box:", results)

            combined_text = ""
            for detection in results:
                text = detection[1]
                if "IND" not in text.upper():
                    combined_text += text.replace(" ", "")

            if 6 <= len(combined_text) <= 14 and any(c.isdigit() for c in combined_text):
                return combined_text


    # Step 2: Fallback to full image OCR
    results = reader.readtext(image_np)
    print("Full image OCR results:", results)


    for detection in results:
        text = detection[1]
        if 6 <= len(text) <= 14 and any(c.isdigit() for c in text) and "IND" not in text.upper():
            return text.upper()
    print("Nothing matched the filtering rules.")
    return "No plate detected"

