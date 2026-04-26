import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image

import mamba_registry
from ultralytics import YOLO


class EigenCAMWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        super().__init__()
        self.model = model
        self._activation = None
        target_layer.register_forward_hook(
            lambda m, inp, out: setattr(self, "_activation", out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model(x)
        act = self._activation
        if act.ndim == 4:
            return act
        return act.unsqueeze(0)  # safety fallback


def get_detections(yolo: YOLO, img_bgr: np.ndarray, conf: float = 0.25):
    """Run YOLO inference and return boxes as [(x1,y1,x2,y2), ...]."""
    results = yolo(img_bgr, verbose=False, conf=conf)
    boxes = []
    for r in results:
        for box in r.boxes.xyxy.cpu().numpy().astype(int):
            boxes.append(tuple(box[:4]))
    return boxes


def draw_detections(img_bgr: np.ndarray, boxes: list) -> np.ndarray:
    out = img_bgr.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


def renormalize_cam_in_boxes(boxes, grayscale_cam: np.ndarray) -> np.ndarray:
    """Zero CAM outside boxes, renormalize [0,1] inside each box."""
    renorm = np.zeros_like(grayscale_cam)
    for x1, y1, x2, y2 in boxes:
        patch = grayscale_cam[y1:y2, x1:x2]
        if patch.size > 0:
            renorm[y1:y2, x1:x2] = scale_cam_image(patch.copy())
    return scale_cam_image(renorm)



def load_image(path: str, imgsz: int = 640):
    """Returns (rgb_float32 [H,W,3], tensor [1,3,H,W], bgr_uint8)."""
    img_bgr = cv2.imread(path)
    img_bgr = cv2.resize(img_bgr, (imgsz, imgsz))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
    return img_float, tensor, img_bgr


def get_image_paths(source: str) -> list[str]:
    p = Path(source)
    if p.is_file():
        return [str(p)]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in exts)



def run(weights: str, source: str, layer_idx: int, out_dir: str,
        imgsz: int = 640, device: str = "cuda", max_images: int = 20,
        conf: float = 0.25):

    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    yolo = YOLO(weights)
    det_model = yolo.model.to(dev).eval()

    target_layer = det_model.model[layer_idx]
    print(f"Target layer [{layer_idx}]: {type(target_layer).__name__}")


    wrapper = EigenCAMWrapper(det_model, target_layer).to(dev).eval()

    cam = EigenCAM(model=wrapper, target_layers=[target_layer])

    image_paths = get_image_paths(source)[:max_images]
    print(f"Processing {len(image_paths)} images → {out_dir}")

    for img_path in image_paths:
        img_float, tensor, img_bgr = load_image(img_path, imgsz)
        tensor = tensor.to(dev)

        # Raw EigenCAM heatmap
        grayscale_cam = cam(input_tensor=tensor)[0]  # [H, W]

        boxes = get_detections(yolo, img_bgr, conf=conf)

        stem = Path(img_path).stem

        # 1) Full-image cam overlay
        vis_full = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        vis_full_bgr = cv2.cvtColor(vis_full, cv2.COLOR_RGB2BGR)

        if boxes:
            # 2) Renormalized cam (inside boxes only)
            renorm_cam = renormalize_cam_in_boxes(boxes, grayscale_cam)
            vis_renorm = show_cam_on_image(img_float, renorm_cam, use_rgb=True)
            vis_renorm_bgr = cv2.cvtColor(vis_renorm, cv2.COLOR_RGB2BGR)
            vis_renorm_bgr = draw_detections(vis_renorm_bgr, boxes)

            # 3) Side-by-side: original | full cam | renorm cam
            side = np.hstack([img_bgr, vis_full_bgr, vis_renorm_bgr])
        else:
            side = np.hstack([img_bgr, vis_full_bgr])

        cv2.imwrite(os.path.join(out_dir, f"{stem}_cam.jpg"), vis_full_bgr)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_compare.jpg"), side)
        if boxes:
            cv2.imwrite(os.path.join(out_dir, f"{stem}_cam_boxes.jpg"), vis_renorm_bgr)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EigenCAM for YOLO + VSSMBlock models")
    parser.add_argument("--weights", required=True, help="Path to YOLO .pt weights")
    parser.add_argument("--source", required=True, help="Image file or directory")
    parser.add_argument("--layer", type=int, default=-6,
                        help="model.model.model[N] to hook (default: -2 = second-to-last)")
    parser.add_argument("--out", default="fig/gradcam", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    args = parser.parse_args()

    run(
        weights=args.weights,
        source=args.source,
        layer_idx=args.layer,
        out_dir=args.out,
        imgsz=args.imgsz,
        device=args.device,
        max_images=args.max_images,
        conf=args.conf,
    )
