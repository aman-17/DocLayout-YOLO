import cv2
import numpy as np
from doclayout_yolo import YOLOv10

def keep_only_abandon_tag(image_path, output_path, conf=0.3, device="cuda:0"):
    """
    Keeps only regions detected with the "abandon" tag and whites out everything else.
    Does not draw rectangles or labels around the preserved regions.
    
    Args:
        image_path: Path to input document image
        output_path: Path to save processed image
        conf: Confidence threshold for detection
        device: Device to run model on ("cuda:0" or "cpu")
    """
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    white_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    model = YOLOv10("./doclayout_yolo_docstructbench_imgsz1024.pt")
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
    keep_only_abandon_tag(
        image_path="./page_7.png",
        output_path="./only_abandon_tag.jpg"
    )
