import boto3
import cv2
import os
import pypdf
import base64
import io
import subprocess
import numpy as np
import argparse
from doclayout_yolo import YOLOv10
from PIL import Image


def download_pdf_from_s3(s3_path: str, local_path: str) -> bool:
    """
    Download a PDF file from S3.

    Args:
        s3_path: The S3 path (s3://bucket/path/to/file.pdf)
        local_path: The local path to save the file

    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        parts = s3_path.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1]

        s3 = boto3.client("s3")

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        s3.download_file(bucket, key, local_path)
        return True
    except Exception as e:
        print(f"Error downloading {s3_path}: {str(e)}")
        return False
    

def extract_page_from_pdf(input_path: str, output_path: str, page_num: int) -> bool:
    """
    Extract a specific page from a PDF and save it as a new PDF.

    Args:
        input_path: Path to the input PDF
        output_path: Path to save the extracted page
        page_num: The page number to extract (0-indexed)

    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        reader = pypdf.PdfReader(input_path)
        if page_num >= len(reader.pages):
            print(f"Page number {page_num} out of range for {input_path} with {len(reader.pages)} pages")
            return False

        writer = pypdf.PdfWriter()
        writer.add_page(reader.pages[page_num])

        with open(output_path, "wb") as output_file:
            writer.write(output_file)

        return True
    except Exception as e:
        print(f"Error extracting page {page_num} from {input_path}: {str(e)}")
        raise


def get_pdf_media_box_width_height(local_pdf_path: str, page_num: int) -> tuple[float, float]:
    """
    Get the MediaBox dimensions for a specific page in a PDF file using the pdfinfo command.

    :param pdf_file: Path to the PDF file
    :param page_num: The page number for which to extract MediaBox dimensions
    :return: A tuple containing MediaBox dimensions (width, height)
    """
    command = ["pdfinfo", "-f", str(page_num), "-l", str(page_num), "-box", "-enc", "UTF-8", local_pdf_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise ValueError(f"Error running pdfinfo: {result.stderr}")
    
    output = result.stdout
    
    for line in output.splitlines():
        if "MediaBox" in line:
            media_box_str = line.split(":")[1].strip().split()
            media_box = [float(x) for x in media_box_str]
            return abs(media_box[0] - media_box[2]), abs(media_box[3] - media_box[1])

    raise ValueError("MediaBox not found in the PDF info.")


def render_pdf_to_png(local_pdf_path: str, output_png_path: str, page_num: int, target_longest_image_dim: int = 2048) -> None:
    """
    Render a PDF page to a PNG file
    
    Args:
        local_pdf_path: Path to the PDF file
        output_png_path: Path to save the PNG image
        page_num: The page number to render (1-indexed for pdftoppm)
        target_longest_image_dim: Target resolution for the longest dimension
    """
    # Get dimensions to calculate appropriate resolution
    longest_dim = max(get_pdf_media_box_width_height(local_pdf_path, page_num))
    
    # Calculate the resolution to achieve the target dimension
    resolution = str(int(target_longest_image_dim * 72 / longest_dim))  # 72 pixels per point
    
    # Convert PDF page to PNG using pdftoppm
    command = [
        "pdftoppm",
        "-png",
        "-singlefile",  # Output a single file
        "-f", str(page_num),
        "-l", str(page_num),
        "-r", resolution,
        local_pdf_path,
        output_png_path.rstrip(".png")  # pdftoppm adds the extension automatically
    ]
    
    subprocess.run(
        command,
        timeout=120,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def process_image_with_model(image_path, model_path, conf=0.3, device="cuda:0"):
    """
    Process an image with the YOLOv10 model to keep only regions with 'abandon' tag.
    
    Args:
        image_path: Path to input document image
        model_path: Path to the YOLOv10 model
        conf: Confidence threshold for detection
        device: Device to run model on ("cuda:0" or "cpu")
        
    Returns:
        Processed image as a numpy array
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    height, width = image.shape[:2]
    white_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Load the model and predict
    model = YOLOv10(model_path)
    det_res = model.predict(
        image_path,
        imgsz=1024,
        conf=conf,
        device=device
    )
    
    # Process detection results
    for result in det_res:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = result.names[cls_id]
            print(f"Detected: {cls_name} (ID: {cls_id})")
            if cls_name == "abandon":
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                white_canvas[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    
    return white_canvas


def image_to_pdf(image_array, output_pdf_path):
    """
    Convert a numpy image array to a PDF file.
    
    Args:
        image_array: Numpy array containing the image
        output_pdf_path: Path where to save the PDF
    """
    # Convert BGR (OpenCV format) to RGB for PIL
    if image_array.shape[2] == 3:  # Check if it's a color image
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Create a PIL Image from the numpy array
    pil_image = Image.fromarray(image_array)
    
    # Save as PDF
    pil_image.save(output_pdf_path, "PDF", resolution=100.0)


def process_pdf(s3_path, temp_dir, output_dir, model_path, conf=0.3, device="cuda:0"):
    """
    Process a single PDF from S3.
    
    Args:
        s3_path: S3 path to the PDF
        temp_dir: Directory for temporary files
        output_dir: Directory to save output files
        model_path: Path to the YOLOv10 model
        conf: Confidence threshold for detection
        device: Device to run model on
    """

    filename = os.path.basename(s3_path)
    base_filename = os.path.splitext(filename)[0]
    local_pdf_path = os.path.join(temp_dir, filename)
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"Downloading {s3_path}...")
        success = download_pdf_from_s3(s3_path, local_pdf_path)
        if not success:
            print(f"Failed to download {s3_path}. Skipping.")
            return
        
        with open(local_pdf_path, "rb") as file:
            pdf = pypdf.PdfReader(file)
            num_pages = len(pdf.pages)
        
        print(f"Processing {filename} with {num_pages} pages")
        
        for page_num in range(num_pages):
            page_pdf_path = os.path.join(temp_dir, f"{base_filename}_page_{page_num+1}.pdf")
            page_png_path = os.path.join(temp_dir, f"{base_filename}_page_{page_num+1}.png")
            output_pdf_path = os.path.join(output_dir, f"{base_filename}_page_{page_num+1}_processed.pdf")
            
            print(f"Extracting page {page_num+1}...")
            extract_page_from_pdf(local_pdf_path, page_pdf_path, page_num)
            
            print(f"Converting page {page_num+1} to PNG...")
            render_pdf_to_png(page_pdf_path, page_png_path, 1)
            
            print(f"Processing page {page_num+1} with model...")
            processed_image = process_image_with_model(page_png_path, model_path, conf, device)
            
            print(f"Converting processed image to PDF...")
            image_to_pdf(processed_image, output_pdf_path)
            
            print(f"Page {page_num+1} processed and saved to {output_pdf_path}")
            
    except Exception as e:
        print(f"Error processing {s3_path}: {str(e)}")
    finally:
        # Clean up temporary files
        for file in os.listdir(temp_dir):
            if file.startswith(base_filename):
                os.remove(os.path.join(temp_dir, file))
        pass


def main():
    parser = argparse.ArgumentParser(description="Process PDFs with 'abandon' tag detection")
    parser.add_argument("--input_list", type=str, required=True, help="Path to text file with S3 paths")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed PDFs")
    parser.add_argument("--model_path", type=str, required=True, help="Path to YOLOv10 model")
    parser.add_argument("--temp_dir", type=str, default="./temp", help="Directory for temporary files")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for detection")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run model on ('cuda:0' or 'cpu')")
    
    args = parser.parse_args()

    os.makedirs(args.temp_dir, exist_ok=True)

    with open(args.input_list, "r") as file:
        s3_paths = [line.strip() for line in file if line.strip()]
    
    for s3_path in s3_paths:
        process_pdf(
            s3_path=s3_path,
            temp_dir=args.temp_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            conf=args.conf,
            device=args.device
        )
    print(f"All PDFs processed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
