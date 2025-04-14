import torch
import numpy as np
import cv2
from PIL import Image
import os

class DummySAM:
    def __init__(self, device):
        self.device = device

    def predict(self, image_rgb):
        h, w, _ = image_rgb.shape
        masks = [
            np.zeros((h, w), dtype=bool),
            np.zeros((h, w), dtype=bool)
        ]
        masks[0][100:200, 100:200] = True
        masks[1][220:320, 220:370] = True
        scores = [0.95, 0.89]
        return masks, scores

class DummyPredictor:
    def __init__(self, sam_model):
        self.model = sam_model

    def set_image(self, image_rgb):
        self.image_rgb = image_rgb

    def predict(self, **kwargs):
        masks, scores = self.model.predict(self.image_rgb)
        return masks, scores, None

sam_model = DummySAM(device="cpu")
predictor = DummyPredictor(sam_model)

# ======= 推理流程入口 =======
def run_sam_on_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        multimask_output=True
    )

    result = []
    for i, mask in enumerate(masks):
        bbox = mask_to_bbox(mask)
        result.append({
            "id": i + 1,
            "bbox": bbox.tolist(),
            "confidence": float(scores[i])
        })

    output_path = save_mask_overlay(image_rgb, masks, scores, image_path)
    return {
        "regions": result,
        "visualization": output_path,
        "image_name": os.path.basename(image_path)
    }

def mask_to_bbox(mask):
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return np.array([0, 0, 0, 0])
    x0 = np.min(x_indices)
    x1 = np.max(x_indices)
    y0 = np.min(y_indices)
    y1 = np.max(y_indices)
    return np.array([x0, y0, x1, y1])

def save_mask_overlay(image_rgb, masks, scores, image_path):
    overlay = image_rgb.copy()
    for i, (mask, score) in enumerate(zip(masks, scores)):
        color = np.random.randint(0, 255, (3,)).tolist()

        # ✅ 修复颜色混合问题（通道分开写入）
        for c in range(3):
            overlay[..., c][mask] = (
                0.6 * overlay[..., c][mask] + 0.4 * color[c]
            ).astype(np.uint8)

        bbox = mask_to_bbox(mask)
        cv2.rectangle(
            overlay,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color=color,
            thickness=2
        )
        cv2.putText(
            overlay,
            f"ID {i+1}",
            (bbox[0], bbox[1] - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=1
        )

    image_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    filename = os.path.basename(image_path)
    output_path = os.path.join("static", "uploaded", "sam_result_" + filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image_bgr)
    return output_path
