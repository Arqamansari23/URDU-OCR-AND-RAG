# import os
# import sys
# import tempfile
# import shutil
# from flask import Flask, render_template, request, jsonify, send_from_directory
# from werkzeug.utils import secure_filename
# import threading
# import uuid

# # Import the functionality from the provided files
# from app_book import setup_models, process_image
# from RAG2 import MultilingualRAG
# import pdf2image

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['OUTPUT_FOLDER'] = 'output'
# app.config['VECTOR_STORE_FOLDER'] = 'vector_store'
# app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size

# # Ensure directories exist
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
# os.makedirs(app.config['VECTOR_STORE_FOLDER'], exist_ok=True)

# # Global variables to store models and RAG system
# recognition_model = None
# detection_model = None
# converter = None
# device = None
# rag_system = None
# current_session_id = None

# def initialize_models():
#     """Initialize OCR models"""
#     global recognition_model, detection_model, converter, device
#     if recognition_model is None:
#         recognition_model, detection_model, converter, device = setup_models()

# def initialize_rag_system(groq_api_key, data_file, session_id):
#     """Initialize RAG system"""
#     global rag_system
#     # Create session-specific vector store path
#     vector_store_path = os.path.join(app.config['VECTOR_STORE_FOLDER'], session_id)
#     os.makedirs(vector_store_path, exist_ok=True)
    
#     # Initialize RAG system with session-specific vector store
#     rag_system = MultilingualRAG(groq_api_key, data_file, vector_store_path)
    
#     # Create vector store
#     rag_system.create_vector_store()
    
#     return rag_system

# def pdf_to_images_range(pdf_path, start_page, end_page, dpi=200):
#     """Convert only selected PDF pages to images"""
#     print(f"Converting PDF pages {start_page} to {end_page} to images...")
#     try:
#         # Convert only the selected page range
#         images = pdf2image.convert_from_path(
#             pdf_path, 
#             dpi=dpi, 
#             first_page=start_page, 
#             last_page=end_page
#         )
#         print(f"Successfully converted {len(images)} pages")
#         return images
#     except Exception as e:
#         print(f"Error converting PDF: {e}")
#         return None

# @app.route('/')
# def index():
#     """Serve the main page"""
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     """Handle file upload"""
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     if file and file.filename.lower().endswith('.pdf'):
#         # Generate a unique session ID
#         session_id = str(uuid.uuid4())
#         session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
#         os.makedirs(session_dir, exist_ok=True)
        
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(session_dir, filename)
#         file.save(filepath)
        
#         global current_session_id
#         current_session_id = session_id
        
#         return jsonify({
#             'success': True,
#             'session_id': session_id,
#             'filename': filename,
#             'message': 'File uploaded successfully'
#         })
    
#     return jsonify({'error': 'Only PDF files are allowed'}), 400

# @app.route('/apply_ocr', methods=['POST'])
# def apply_ocr():
#     """Apply OCR on selected pages"""
#     data = request.json
#     session_id = data.get('session_id')
#     start_page = int(data.get('start_page', 1))
#     end_page = int(data.get('end_page', 10))
    
#     if not session_id or session_id != current_session_id:
#         return jsonify({'error': 'Invalid session'}), 400
    
#     session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
#     pdf_files = [f for f in os.listdir(session_dir) if f.lower().endswith('.pdf')]
    
#     if not pdf_files:
#         return jsonify({'error': 'No PDF file found'}), 400
    
#     pdf_path = os.path.join(session_dir, pdf_files[0])
    
#     # Initialize models if not already done
#     initialize_models()
    
#     # Convert only selected pages to images
#     images = pdf_to_images_range(pdf_path, start_page, end_page)
#     if images is None:
#         return jsonify({'error': 'Error converting PDF to images'}), 500
    
#     # Process selected pages
#     extracted_text = []
#     for i, image in enumerate(images):
#         try:
#             page_text = process_image(image, recognition_model, detection_model, converter, device)
#             extracted_text.append(f"=== Page {start_page + i} ===\n{page_text}")
#         except Exception as e:
#             extracted_text.append(f"=== Page {start_page + i} ===\nError processing page: {str(e)}")
    
#     # Save extracted text to file
#     output_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{session_id}.txt")
#     with open(output_file, "w", encoding="utf-8") as f:
#         f.write("\n\n".join(extracted_text))
    
#     return jsonify({
#         'success': True,
#         'text': "\n\n".join(extracted_text),
#         'output_file': f"{session_id}.txt"
#     })

# @app.route('/configure_rag', methods=['POST'])
# def configure_rag():
#     """Configure RAG system"""
#     data = request.json
#     session_id = data.get('session_id')
#     groq_api_key = data.get('groq_api_key')
#     start_page = int(data.get('start_page', 1))
#     end_page = int(data.get('end_page', 10))
    
#     if not session_id or session_id != current_session_id:
#         return jsonify({'error': 'Invalid session'}), 400
    
#     if not groq_api_key:
#         return jsonify({'error': 'Groq API key is required'}), 400
    
#     session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
#     pdf_files = [f for f in os.listdir(session_dir) if f.lower().endswith('.pdf')]
    
#     if not pdf_files:
#         return jsonify({'error': 'No PDF file found'}), 400
    
#     pdf_path = os.path.join(session_dir, pdf_files[0])
    
#     # Initialize models if not already done
#     initialize_models()
    
#     # Convert only selected pages to images
#     images = pdf_to_images_range(pdf_path, start_page, end_page)
#     if images is None:
#         return jsonify({'error': 'Error converting PDF to images'}), 500
    
#     # Process selected pages
#     extracted_text = []
#     for i, image in enumerate(images):
#         try:
#             page_text = process_image(image, recognition_model, detection_model, converter, device)
#             extracted_text.append(f"=== Page {start_page + i} ===\n{page_text}")
#         except Exception as e:
#             extracted_text.append(f"=== Page {start_page + i} ===\nError processing page: {str(e)}")
    
#     # Save extracted text to file
#     output_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{session_id}.txt")
#     with open(output_file, "w", encoding="utf-8") as f:
#         f.write("\n\n".join(extracted_text))
    
#     # Initialize RAG system
#     try:
#         initialize_rag_system(groq_api_key, output_file, session_id)
#         return jsonify({
#             'success': True,
#             'message': 'RAG system configured successfully',
#             'output_file': f"{session_id}.txt",
#             'session_id': session_id
#         })
#     except Exception as e:
#         return jsonify({'error': f'Error configuring RAG system: {str(e)}'}), 500

# @app.route('/chat', methods=['POST'])
# def chat():
#     """Handle chat queries"""
#     data = request.json
#     query = data.get('query')
#     session_id = data.get('session_id')
    
#     if not query:
#         return jsonify({'error': 'Query is required'}), 400
    
#     if not session_id or session_id != current_session_id:
#         return jsonify({'error': 'Invalid session'}), 400
    
#     if rag_system is None:
#         return jsonify({'error': 'RAG system not configured'}), 400
    
#     try:
#         response = rag_system.answer_query(query)
#         return jsonify({
#             'success': True,
#             'response': response
#         })
#     except Exception as e:
#         return jsonify({'error': f'Error processing query: {str(e)}'}), 500

# @app.route('/output/<filename>')
# def download_output(filename):
#     """Download output file"""
#     return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)



# import os
# import sys
# import tempfile
# import shutil
# from flask import Flask, render_template, request, jsonify, send_from_directory
# from werkzeug.utils import secure_filename
# import threading
# import uuid
# import logging

# # Import the functionality from the provided files
# from app_book import setup_models, process_image
# from RAG2 import MultilingualRAG
# import pdf2image

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['OUTPUT_FOLDER'] = 'output'
# app.config['VECTOR_STORE_FOLDER'] = 'vector_store'
# app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size

# # Ensure directories exist
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
# os.makedirs(app.config['VECTOR_STORE_FOLDER'], exist_ok=True)

# # Global variables to store models and RAG system
# recognition_model = None
# detection_model = None
# converter = None
# device = None
# rag_system = None
# current_session_id = None

# def initialize_models():
#     """Initialize OCR models"""
#     global recognition_model, detection_model, converter, device
#     if recognition_model is None:
#         recognition_model, detection_model, converter, device = setup_models()

# def initialize_rag_system(groq_api_key, data_file, session_id):
#     """Initialize RAG system"""
#     global rag_system
#     # Create session-specific vector store path
#     vector_store_path = os.path.join(app.config['VECTOR_STORE_FOLDER'], session_id)
#     logger.info(f"Creating vector store path: {vector_store_path}")
#     os.makedirs(vector_store_path, exist_ok=True)
    
#     # Initialize RAG system with session-specific vector store
#     rag_system = MultilingualRAG(groq_api_key, data_file, vector_store_path)
    
#     # Create vector store
#     logger.info("Creating vector store...")
#     rag_system.create_vector_store()
#     logger.info("Vector store created successfully")
    
#     # Verify vector store files exist
#     faiss_index_file = os.path.join(vector_store_path, "faiss_index.bin")
#     documents_file = os.path.join(vector_store_path, "documents.pkl")
#     chunks_file = os.path.join(vector_store_path, "chunks.pkl")
    
#     if not os.path.exists(faiss_index_file):
#         raise Exception(f"FAISS index file not created at {faiss_index_file}")
#     if not os.path.exists(documents_file):
#         raise Exception(f"Documents file not created at {documents_file}")
#     if not os.path.exists(chunks_file):
#         raise Exception(f"Chunks file not created at {chunks_file}")
    
#     logger.info("All vector store files verified")
#     return rag_system

# def pdf_to_images_range(pdf_path, start_page, end_page, dpi=200):
#     """Convert only selected PDF pages to images"""
#     logger.info(f"Converting PDF pages {start_page} to {end_page} to images...")
#     try:
#         # Convert only the selected page range
#         images = pdf2image.convert_from_path(
#             pdf_path, 
#             dpi=dpi, 
#             first_page=start_page, 
#             last_page=end_page
#         )
#         logger.info(f"Successfully converted {len(images)} pages")
#         return images
#     except Exception as e:
#         logger.error(f"Error converting PDF: {e}")
#         return None

# @app.route('/')
# def index():
#     """Serve the main page"""
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     """Handle file upload"""
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     if file and file.filename.lower().endswith('.pdf'):
#         # Generate a unique session ID
#         session_id = str(uuid.uuid4())
#         session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
#         os.makedirs(session_dir, exist_ok=True)
        
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(session_dir, filename)
#         file.save(filepath)
        
#         global current_session_id
#         current_session_id = session_id
        
#         logger.info(f"File uploaded: {filename}, session_id: {session_id}")
        
#         return jsonify({
#             'success': True,
#             'session_id': session_id,
#             'filename': filename,
#             'message': 'File uploaded successfully'
#         })
    
#     return jsonify({'error': 'Only PDF files are allowed'}), 400

# @app.route('/apply_ocr', methods=['POST'])
# def apply_ocr():
#     """Apply OCR on selected pages"""
#     data = request.json
#     session_id = data.get('session_id')
#     start_page = int(data.get('start_page', 1))
#     end_page = int(data.get('end_page', 10))
    
#     if not session_id or session_id != current_session_id:
#         return jsonify({'error': 'Invalid session'}), 400
    
#     session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
#     pdf_files = [f for f in os.listdir(session_dir) if f.lower().endswith('.pdf')]
    
#     if not pdf_files:
#         return jsonify({'error': 'No PDF file found'}), 400
    
#     pdf_path = os.path.join(session_dir, pdf_files[0])
    
#     # Initialize models if not already done
#     initialize_models()
    
#     # Convert only selected pages to images
#     images = pdf_to_images_range(pdf_path, start_page, end_page)
#     if images is None:
#         return jsonify({'error': 'Error converting PDF to images'}), 500
    
#     # Process selected pages
#     extracted_text = []
#     for i, image in enumerate(images):
#         try:
#             page_text = process_image(image, recognition_model, detection_model, converter, device)
#             extracted_text.append(f"=== Page {start_page + i} ===\n{page_text}")
#         except Exception as e:
#             extracted_text.append(f"=== Page {start_page + i} ===\nError processing page: {str(e)}")
    
#     # Save extracted text to file
#     output_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{session_id}.txt")
#     with open(output_file, "w", encoding="utf-8") as f:
#         f.write("\n\n".join(extracted_text))
    
#     logger.info(f"OCR completed for session {session_id}, pages {start_page}-{end_page}")
    
#     return jsonify({
#         'success': True,
#         'text': "\n\n".join(extracted_text),
#         'output_file': f"{session_id}.txt"
#     })

# @app.route('/configure_rag', methods=['POST'])
# def configure_rag():
#     """Configure RAG system"""
#     data = request.json
#     session_id = data.get('session_id')
#     groq_api_key = data.get('groq_api_key')
#     start_page = int(data.get('start_page', 1))
#     end_page = int(data.get('end_page', 10))
    
#     if not session_id or session_id != current_session_id:
#         return jsonify({'error': 'Invalid session'}), 400
    
#     if not groq_api_key:
#         return jsonify({'error': 'Groq API key is required'}), 400
    
#     session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
#     pdf_files = [f for f in os.listdir(session_dir) if f.lower().endswith('.pdf')]
    
#     if not pdf_files:
#         return jsonify({'error': 'No PDF file found'}), 400
    
#     pdf_path = os.path.join(session_dir, pdf_files[0])
    
#     # Initialize models if not already done
#     initialize_models()
    
#     # Convert only selected pages to images
#     images = pdf_to_images_range(pdf_path, start_page, end_page)
#     if images is None:
#         return jsonify({'error': 'Error converting PDF to images'}), 500
    
#     # Process selected pages
#     extracted_text = []
#     for i, image in enumerate(images):
#         try:
#             page_text = process_image(image, recognition_model, detection_model, converter, device)
#             extracted_text.append(f"=== Page {start_page + i} ===\n{page_text}")
#         except Exception as e:
#             extracted_text.append(f"=== Page {start_page + i} ===\nError processing page: {str(e)}")
    
#     # Save extracted text to file
#     output_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{session_id}.txt")
#     with open(output_file, "w", encoding="utf-8") as f:
#         f.write("\n\n".join(extracted_text))
    
#     # Initialize RAG system
#     try:
#         logger.info(f"Initializing RAG system for session {session_id}")
#         initialize_rag_system(groq_api_key, output_file, session_id)
        
#         # Verify vector store directory exists and has files
#         vector_store_path = os.path.join(app.config['VECTOR_STORE_FOLDER'], session_id)
#         if not os.path.exists(vector_store_path):
#             raise Exception(f"Vector store directory not created: {vector_store_path}")
        
#         files = os.listdir(vector_store_path)
#         if not files:
#             raise Exception(f"Vector store directory is empty: {vector_store_path}")
        
#         logger.info(f"RAG system configured successfully for session {session_id}")
#         return jsonify({
#             'success': True,
#             'message': 'RAG system configured successfully',
#             'output_file': f"{session_id}.txt",
#             'session_id': session_id,
#             'vector_store_files': files
#         })
#     except Exception as e:
#         logger.error(f"Error configuring RAG system: {str(e)}")
#         return jsonify({'error': f'Error configuring RAG system: {str(e)}'}), 500

# @app.route('/chat', methods=['POST'])
# def chat():
#     """Handle chat queries"""
#     data = request.json
#     query = data.get('query')
#     session_id = data.get('session_id')
    
#     if not query:
#         return jsonify({'error': 'Query is required'}), 400
    
#     if not session_id or session_id != current_session_id:
#         return jsonify({'error': 'Invalid session'}), 400
    
#     if rag_system is None:
#         return jsonify({'error': 'RAG system not configured'}), 400
    
#     try:
#         logger.info(f"Processing chat query: {query}")
#         response = rag_system.answer_query(query)
#         return jsonify({
#             'success': True,
#             'response': response
#         })
#     except Exception as e:
#         logger.error(f"Error processing chat query: {str(e)}")
#         return jsonify({'error': f'Error processing query: {str(e)}'}), 500

# @app.route('/output/<filename>')
# def download_output(filename):
#     """Download output file"""
#     return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)



import os
import sys
import tempfile
import shutil
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import threading
import uuid
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix for MPS and frombuffer errors - add this before any torch operations
# Check if MPS is available and handle the error gracefully
try:
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except AttributeError:
    # MPS is not available in this PyTorch build
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Using device: {device}")

# Monkey patch torch.frombuffer if it doesn't exist
if not hasattr(torch, 'frombuffer'):
    def frombuffer_patch(buffer, dtype, *args, **kwargs):
        # Convert buffer to numpy array first, then to tensor
        np_array = np.frombuffer(buffer, dtype=np.float32)
        return torch.from_numpy(np_array)
    
    torch.frombuffer = frombuffer_patch
    logger.info("Applied monkey patch for torch.frombuffer")

# Import the functionality from the provided files
from app_book import setup_models, process_image
from RAG2 import MultilingualRAG
import pdf2image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['VECTOR_STORE_FOLDER'] = 'vector_store'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['VECTOR_STORE_FOLDER'], exist_ok=True)

# Global variables to store models and RAG system
recognition_model = None
detection_model = None
converter = None
model_device = None
rag_system = None
current_session_id = None

def initialize_models():
    """Initialize OCR models"""
    global recognition_model, detection_model, converter, model_device
    if recognition_model is None:
        recognition_model, detection_model, converter, model_device = setup_models()

def initialize_rag_system(groq_api_key, data_file, session_id):
    """Initialize RAG system"""
    global rag_system
    # Create session-specific vector store path
    vector_store_path = os.path.join(app.config['VECTOR_STORE_FOLDER'], session_id)
    logger.info(f"Creating vector store path: {vector_store_path}")
    os.makedirs(vector_store_path, exist_ok=True)
    
    # Initialize RAG system with session-specific vector store
    rag_system = MultilingualRAG(groq_api_key, data_file, vector_store_path)
    
    # Create vector store
    logger.info("Creating vector store...")
    rag_system.create_vector_store()
    logger.info("Vector store created successfully")
    
    # Verify vector store files exist
    faiss_index_file = os.path.join(vector_store_path, "faiss_index.bin")
    documents_file = os.path.join(vector_store_path, "documents.pkl")
    chunks_file = os.path.join(vector_store_path, "chunks.pkl")
    
    if not os.path.exists(faiss_index_file):
        raise Exception(f"FAISS index file not created at {faiss_index_file}")
    if not os.path.exists(documents_file):
        raise Exception(f"Documents file not created at {documents_file}")
    if not os.path.exists(chunks_file):
        raise Exception(f"Chunks file not created at {chunks_file}")
    
    logger.info("All vector store files verified")
    return rag_system

def pdf_to_images_range(pdf_path, start_page, end_page, dpi=200):
    """Convert only selected PDF pages to images"""
    logger.info(f"Converting PDF pages {start_page} to {end_page} to images...")
    try:
        # Convert only the selected page range
        images = pdf2image.convert_from_path(
            pdf_path, 
            dpi=dpi, 
            first_page=start_page, 
            last_page=end_page
        )
        logger.info(f"Successfully converted {len(images)} pages")
        return images
    except Exception as e:
        logger.error(f"Error converting PDF: {e}")
        return None

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(session_dir, filename)
        file.save(filepath)
        
        global current_session_id
        current_session_id = session_id
        
        logger.info(f"File uploaded: {filename}, session_id: {session_id}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'filename': filename,
            'message': 'File uploaded successfully'
        })
    
    return jsonify({'error': 'Only PDF files are allowed'}), 400

@app.route('/apply_ocr', methods=['POST'])
def apply_ocr():
    """Apply OCR on selected pages"""
    data = request.json
    session_id = data.get('session_id')
    start_page = int(data.get('start_page', 1))
    end_page = int(data.get('end_page', 10))
    
    if not session_id or session_id != current_session_id:
        return jsonify({'error': 'Invalid session'}), 400
    
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    pdf_files = [f for f in os.listdir(session_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        return jsonify({'error': 'No PDF file found'}), 400
    
    pdf_path = os.path.join(session_dir, pdf_files[0])
    
    # Initialize models if not already done
    initialize_models()
    
    # Convert only selected pages to images
    images = pdf_to_images_range(pdf_path, start_page, end_page)
    if images is None:
        return jsonify({'error': 'Error converting PDF to images'}), 500
    
    # Process selected pages
    extracted_text = []
    for i, image in enumerate(images):
        try:
            page_text = process_image(image, recognition_model, detection_model, converter, model_device)
            extracted_text.append(f"=== Page {start_page + i} ===\n{page_text}")
        except Exception as e:
            extracted_text.append(f"=== Page {start_page + i} ===\nError processing page: {str(e)}")
    
    # Save extracted text to file
    output_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{session_id}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(extracted_text))
    
    logger.info(f"OCR completed for session {session_id}, pages {start_page}-{end_page}")
    
    return jsonify({
        'success': True,
        'text': "\n\n".join(extracted_text),
        'output_file': f"{session_id}.txt"
    })

@app.route('/configure_rag', methods=['POST'])
def configure_rag():
    """Configure RAG system"""
    data = request.json
    session_id = data.get('session_id')
    groq_api_key = data.get('groq_api_key')
    start_page = int(data.get('start_page', 1))
    end_page = int(data.get('end_page', 10))
    
    if not session_id or session_id != current_session_id:
        return jsonify({'error': 'Invalid session'}), 400
    
    if not groq_api_key:
        return jsonify({'error': 'Groq API key is required'}), 400
    
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    pdf_files = [f for f in os.listdir(session_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        return jsonify({'error': 'No PDF file found'}), 400
    
    pdf_path = os.path.join(session_dir, pdf_files[0])
    
    # Initialize models if not already done
    initialize_models()
    
    # Convert only selected pages to images
    images = pdf_to_images_range(pdf_path, start_page, end_page)
    if images is None:
        return jsonify({'error': 'Error converting PDF to images'}), 500
    
    # Process selected pages
    extracted_text = []
    for i, image in enumerate(images):
        try:
            page_text = process_image(image, recognition_model, detection_model, converter, model_device)
            extracted_text.append(f"=== Page {start_page + i} ===\n{page_text}")
        except Exception as e:
            extracted_text.append(f"=== Page {start_page + i} ===\nError processing page: {str(e)}")
    
    # Save extracted text to file
    output_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{session_id}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(extracted_text))
    
    # Initialize RAG system
    try:
        logger.info(f"Initializing RAG system for session {session_id}")
        initialize_rag_system(groq_api_key, output_file, session_id)
        
        # Verify vector store directory exists and has files
        vector_store_path = os.path.join(app.config['VECTOR_STORE_FOLDER'], session_id)
        if not os.path.exists(vector_store_path):
            raise Exception(f"Vector store directory not created: {vector_store_path}")
        
        files = os.listdir(vector_store_path)
        if not files:
            raise Exception(f"Vector store directory is empty: {vector_store_path}")
        
        logger.info(f"RAG system configured successfully for session {session_id}")
        return jsonify({
            'success': True,
            'message': 'RAG system configured successfully',
            'output_file': f"{session_id}.txt",
            'session_id': session_id,
            'vector_store_files': files
        })
    except Exception as e:
        logger.error(f"Error configuring RAG system: {str(e)}")
        return jsonify({'error': f'Error configuring RAG system: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat queries"""
    data = request.json
    query = data.get('query')
    session_id = data.get('session_id')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    if not session_id or session_id != current_session_id:
        return jsonify({'error': 'Invalid session'}), 400
    
    if rag_system is None:
        return jsonify({'error': 'RAG system not configured'}), 400
    
    try:
        logger.info(f"Processing chat query: {query}")
        response = rag_system.answer_query(query)
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        logger.error(f"Error processing chat query: {str(e)}")
        return jsonify({'error': f'Error processing query: {str(e)}'}), 500

@app.route('/output/<filename>')
def download_output(filename):
    """Download output file"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)