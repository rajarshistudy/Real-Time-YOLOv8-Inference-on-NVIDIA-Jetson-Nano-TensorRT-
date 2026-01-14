import cv2
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def yolo_xywh_to_xyxy_pixels(x, y, w, h):
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.stack((x1, y1, x2, y2), axis=1)

    

def post_process(output, img_w, img_h, conf_thresh=0.25, iou_thresh=0.45):
    preds = output[0]   # This gives us (84, 8400)
    box_coord = preds[0:4, :]   # x, y, w, h
    class_scores = preds[4:, :] # (80, 8400)

    # Compute the max confidence for each of the 8400 columns
    # Idea here is to collapse the 80 rows to one maximum so axis = 0
    scores = np.max(class_scores, axis=0)
    class_ids = np.argmax(class_scores, axis=0)
    # Small sanity check for shapes
    assert scores.shape[0] == class_ids.shape[0] == 8400

    # Filtering out confidence predictions
    mask = scores > conf_thresh
    # Discards any prediction below the confidence threshold 
    box_coord = box_coord[:, mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    x, y, w, h = box_coord[0], box_coord[1], box_coord[2], box_coord[3]
    assert x.shape == y.shape == w.shape == h.shape
    boxes_xyxy = yolo_xywh_to_xyxy_pixels(x, y, w, h)
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, img_w - 1)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, img_h - 1)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, img_w - 1)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, img_h - 1)
    
    # Converting boxes to NMS compatability
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]

    x = x1
    y = y1
    w = x2-x1
    h = y2-y1

    boxes = np.stack((x, y, w, h), axis=1)

    # Use cv2.dnn.NMSBoxes to compute the NMS compression - note that this takes lists as inputs
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thresh, iou_thresh, eta=1.0, top_k=0)

    # Flatten the indices given from NMS compression
    if len(indices) > 0:
        indices = indices.flatten()
        # Store the final boxes, scores and class IDs to draw the rectangles
        final_boxes = []
        final_scores = []
        final_class_ids = []
        for i in indices:
            final_boxes.append(boxes_xyxy[i])
            final_scores.append(scores[i])
            final_class_ids.append(class_ids[i])
        
        assert len(final_boxes) == len(final_scores) == len(final_class_ids)
    else:
        final_boxes = []
        final_scores = []
        final_class_ids = []
    return final_boxes, final_scores, final_class_ids

    