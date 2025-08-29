import torch
import os
import sys
from pathlib import Path
from PIL import Image
import pdf2image
from read import text_recognizer
from model import Model
from utils import CTCLabelConverter
from ultralytics import YOLO

# Configuration
MAX_PAGES_TO_EXTRACT = 10  # Number of pages to process

def setup_models():
    """Initialize OCR models"""
    print("Loading models...")
    
    # Load vocabulary
    with open("UrduGlyphs.txt", "r", encoding="utf-8") as file:
        content = file.readlines()
    content = ''.join([str(elem).strip('\n') for elem in content])
    content = content + " "
    
    # Model configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    converter = CTCLabelConverter(content)
    recognition_model = Model(num_class=len(converter.character), device=device)
    recognition_model = recognition_model.to(device)
    recognition_model.load_state_dict(torch.load("best_norm_ED.pth", map_location=device))
    recognition_model.eval()
    
    detection_model = YOLO("yolov8m_UrduDoc.pt")
    
    print("Models loaded successfully!")
    return recognition_model, detection_model, converter, device

def process_image(image, recognition_model, detection_model, converter, device):
    """Process a single image and extract Urdu text"""
    # Line Detection
    detection_results = detection_model.predict(
        source=image, 
        conf=0.2, 
        imgsz=1280, 
        save=False, 
        nms=True, 
        device=device,
        verbose=False
    )
    
    if len(detection_results[0].boxes) == 0:
        return "No text detected in this image."
    
    bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
    bounding_boxes.sort(key=lambda x: x[1])  # Sort by y-coordinate (top to bottom)
    
    # Crop the detected lines
    cropped_images = []
    for box in bounding_boxes:
        cropped_images.append(image.crop(box))
    
    # Recognize the text
    texts = []
    for img in cropped_images:
        text = text_recognizer(img, recognition_model, converter, device)
        texts.append(text)
    
    # Join the text
    return "\n".join(texts)

def pdf_to_images(pdf_path, dpi=200):
    """Convert PDF pages to images"""
    print(f"Converting PDF to images...")
    try:
        images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
        print(f"Successfully converted {len(images)} pages")
        return images
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return None

def process_pdf(pdf_path, output_dir="output"):
    """Main function to process PDF and extract Urdu text"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "output1.txt")
    
    # Setup models
    recognition_model, detection_model, converter, device = setup_models()
    
    # Convert PDF to images
    images = pdf_to_images(pdf_path)
    if images is None:
        return
    
    # Limit to MAX_PAGES_TO_EXTRACT pages
    total_pages = len(images)
    pages_to_process = min(MAX_PAGES_TO_EXTRACT, total_pages)
    images = images[:pages_to_process]
    
    print(f"Total pages in PDF: {total_pages}")
    print(f"Processing first {pages_to_process} pages (limit set to {MAX_PAGES_TO_EXTRACT})")
    
    # Process each page
    all_text = []
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i, image in enumerate(images, 1):
            print(f"Processing page {i}/{pages_to_process}...")
            
            try:
                # Process the image
                page_text = process_image(image, recognition_model, detection_model, converter, device)
                
                # Write to file immediately
                f.write(f"=== Page {i} ===\n")
                f.write(page_text)
                f.write(f"\n{'='*50}\n\n")
                f.flush()  # Ensure data is written immediately
                
                print(f"Page {i} processed successfully")
                
            except Exception as e:
                error_msg = f"Error processing page {i}: {e}"
                print(error_msg)
                f.write(f"=== Page {i} ===\n")
                f.write(error_msg)
                f.write(f"\n{'='*50}\n\n")
                f.flush()
    
    print(f"\nProcessing complete! Processed {pages_to_process} out of {total_pages} pages.")
    print(f"Text saved to: {output_file}")

def main():
    """Command line interface"""
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_pdf>")
        print("Example: python script.py book.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Check if PDF file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found!")
        sys.exit(1)
    
    # Check if it's a PDF file
    if not pdf_path.lower().endswith('.pdf'):
        print("Error: Please provide a PDF file!")
        sys.exit(1)
    
    print(f"Starting OCR processing for: {pdf_path}")
    print("This may take a while depending on the size of the PDF...")
    
    try:
        process_pdf(pdf_path)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()