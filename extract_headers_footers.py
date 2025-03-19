import cv2
import numpy as np
import argparse
from doclayout_yolo import YOLOv10

def keep_only_abandon_tag(image_path, output_path, model_path, conf=0.3, device="cuda:0"):
    """
    Keeps only regions detected with the "abandon" tag and whites out everything else.
    Does not draw rectangles or labels around the preserved regions.
    
    Args:
        image_path: Path to input document image
        output_path: Path to save processed image
        model_path: Path to the YOLOv10 model
        conf: Confidence threshold for detection
        device: Device to run model on ("cuda:0" or "cpu")
    """
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    white_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    model = YOLOv10(model_path)
    det_res = model.predict(
        image_path,
        imgsz=1024,
        conf=conf,
        device=device
    )
    
    for result in det_res:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = result.names[cls_id]
            print(cls_name, cls_id)
            if cls_name == "abandon":
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                white_canvas[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    
    cv2.imwrite(output_path, white_canvas)
    print(f"Processed image saved to {output_path}")
    return white_canvas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keep only regions with 'abandon' tag in document images")
    parser.add_argument("--image_path", type=str, default="./page_7.png", help="Path to input document image")
    parser.add_argument("--output_path", type=str, default="./output.jpg", help="Path to save processed image")
    parser.add_argument("--model_path", type=str, default="./doclayout_yolo_docstructbench_imgsz1024.pt", help="Path to YOLOv10 model")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for detection")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run model on ('cuda:0' or 'cpu')")
    
    args = parser.parse_args()
    
    keep_only_abandon_tag(
        image_path=args.image_path,
        output_path=args.output_path,
        model_path=args.model_path,
        conf=args.conf,
        device=args.device
    )