import os
import sys
import tempfile
import shutil
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import torch
from pathlib import Path
from PIL import Image
import pdf2image
from read import text_recognizer
from model import Model
from utils import CTCLabelConverter
from ultralytics import YOLO
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import re
from dotenv import load_dotenv
from typing import List
import gc
import threading
from queue import Queue
import psutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['VECTOR_STORE_FOLDER'] = 'vector_store'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['VECTOR_STORE_FOLDER'], exist_ok=True)

# Global variables to store models and RAG system
recognition_model = None
detection_model = None
converter = None
device = None
rag_system = None

# Status queues for real-time updates
ocr_status_queue = Queue()
rag_status_queue = Queue()

def setup_models():
    """Initialize OCR models"""
    global recognition_model, detection_model, converter, device
    
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

def clear_status_queues():
    """Clear all status queues"""
    while not ocr_status_queue.empty():
        try:
            ocr_status_queue.get_nowait()
        except:
            pass
    
    while not rag_status_queue.empty():
        try:
            rag_status_queue.get_nowait()
        except:
            pass

def get_optimal_batch_size():
    """Dynamically calculate optimal batch size based on available memory"""
    try:
        # Get available memory in GB
        available_memory = psutil.virtual_memory().available / (1024**3)
        
        # Conservative batch size calculation
        # Assume each page uses ~20MB at 200 DPI
        if available_memory > 8:
            return 10  # For systems with >8GB RAM
        elif available_memory > 4:
            return 5   # For systems with >4GB RAM
        else:
            return 2   # For systems with limited RAM
    except:
        return 3  # Default safe batch size

def pdf_to_images_batch(pdf_path, start_page, end_page, batch_size=None, dpi=200):
    """Convert PDF pages to images in batches to manage memory"""
    if batch_size is None:
        batch_size = get_optimal_batch_size()
    
    try:
        # Get total page count first
        with open(pdf_path, 'rb') as f:
            import PyPDF2
            try:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)
            except:
                # Fallback: convert first page to get total count
                temp_images = pdf2image.convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
                del temp_images
                # Use pdf2image to get total pages
                info = pdf2image.pdfinfo_from_path(pdf_path)
                total_pages = info['Pages']
    except:
        # If we can't get page count, proceed with conversion
        total_pages = end_page
    
    # Adjust end_page if necessary
    if end_page > total_pages:
        end_page = total_pages
    
    status_message = f"Converting PDF pages {start_page}-{end_page} in batches of {batch_size}..."
    ocr_status_queue.put(status_message)
    rag_status_queue.put(status_message)
    print(status_message)
    
    # Process in batches
    all_images = []
    
    for batch_start in range(start_page, end_page + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, end_page)
        
        status_message = f"Loading batch: pages {batch_start}-{batch_end}"
        ocr_status_queue.put(status_message)
        rag_status_queue.put(status_message)
        print(status_message)
        
        try:
            # Convert batch of pages
            batch_images = pdf2image.convert_from_path(
                pdf_path, 
                dpi=dpi,
                first_page=batch_start,
                last_page=batch_end
            )
            
            all_images.extend(batch_images)
            
            status_message = f"Loaded {len(batch_images)} pages from batch {batch_start}-{batch_end}"
            ocr_status_queue.put(status_message)
            rag_status_queue.put(status_message)
            print(status_message)
            
        except Exception as e:
            error_msg = f"Error converting batch {batch_start}-{batch_end}: {e}"
            ocr_status_queue.put(error_msg)
            rag_status_queue.put(error_msg)
            print(error_msg)
            continue
    
    status_message = f"Successfully converted {len(all_images)} total pages"
    ocr_status_queue.put(status_message)
    rag_status_queue.put(status_message)
    print(status_message)
    
    return all_images

def process_image_with_memory_management(image, recognition_model, detection_model, converter, device, actual_page_num):
    """Process a single image with memory management"""
    status_message = f"Processing page {actual_page_num}..."
    ocr_status_queue.put(status_message)
    rag_status_queue.put(status_message)
    print(status_message)
    
    try:
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
            status_message = f"Page {actual_page_num}: No text detected"
            ocr_status_queue.put(status_message)
            rag_status_queue.put(status_message)
            return "No text detected in this image."
        
        bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
        bounding_boxes.sort(key=lambda x: x[1])  # Sort by y-coordinate (top to bottom)
        
        # Process lines in smaller batches to manage memory
        texts = []
        line_batch_size = 10  # Process 10 lines at a time
        
        for i in range(0, len(bounding_boxes), line_batch_size):
            batch_boxes = bounding_boxes[i:i + line_batch_size]
            
            # Crop the detected lines for this batch
            cropped_images = []
            for box in batch_boxes:
                cropped_images.append(image.crop(box))
            
            # Recognize the text for this batch
            for img in cropped_images:
                text = text_recognizer(img, recognition_model, converter, device)
                texts.append(text)
            
            # Clear cropped images from memory
            del cropped_images
            
            # Force garbage collection periodically
            if i % (line_batch_size * 3) == 0:
                gc.collect()
        
        # Join the text
        result = "\n".join(texts)
        status_message = f"Page {actual_page_num} processed successfully ({len(texts)} lines)"
        ocr_status_queue.put(status_message)
        rag_status_queue.put(status_message)
        print(status_message)
        
        return result
        
    except Exception as e:
        error_msg = f"Error processing page {actual_page_num}: {e}"
        ocr_status_queue.put(error_msg)
        rag_status_queue.put(error_msg)
        print(error_msg)
        return f"Error processing page: {e}"
    finally:
        # Force garbage collection after each page
        gc.collect()

def process_pdf_batch(pdf_path, start_page=1, end_page=None, output_file="output.txt", batch_size=None):
    """Process PDF with batch processing for memory efficiency"""
    # Setup models if not already loaded
    global recognition_model, detection_model, converter, device
    if recognition_model is None:
        setup_models()
    
    if batch_size is None:
        batch_size = get_optimal_batch_size()
    
    # Get total page count and validate range
    try:
        info = pdf2image.pdfinfo_from_path(pdf_path)
        total_pages = info['Pages']
        
        if end_page is None or end_page > total_pages:
            end_page = total_pages
            
        # Ensure valid range
        start_page = max(1, start_page)
        end_page = min(total_pages, end_page)
        
    except Exception as e:
        error_msg = f"Error getting PDF info: {e}"
        ocr_status_queue.put(error_msg)
        rag_status_queue.put(error_msg)
        print(error_msg)
        return None
    
    status_message = f"Starting batch processing for pages {start_page}-{end_page} (Total: {total_pages} pages)"
    ocr_status_queue.put(status_message)
    rag_status_queue.put(status_message)
    print(status_message)
    
    all_text = []
    
    # Process PDF in page batches
    for batch_start in range(start_page, end_page + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, end_page)
        
        status_message = f"Processing batch: pages {batch_start}-{batch_end}"
        ocr_status_queue.put(status_message)
        rag_status_queue.put(status_message)
        print(status_message)
        
        # Convert current batch to images
        try:
            batch_images = pdf2image.convert_from_path(
                pdf_path, 
                dpi=200,
                first_page=batch_start,
                last_page=batch_end
            )
        except Exception as e:
            error_msg = f"Error converting batch {batch_start}-{batch_end}: {e}"
            ocr_status_queue.put(error_msg)
            rag_status_queue.put(error_msg)
            print(error_msg)
            # Add error placeholders for this batch
            for page_num in range(batch_start, batch_end + 1):
                all_text.append(f"=== Page {page_num} ===\nError processing page: {e}")
            continue
        
        # Process each image in the batch
        for i, image in enumerate(batch_images):
            actual_page_num = batch_start + i
            
            try:
                page_text = process_image_with_memory_management(
                    image, recognition_model, detection_model, converter, device, actual_page_num
                )
                all_text.append(f"=== Page {actual_page_num} ===\n{page_text}")
                
            except Exception as e:
                error_msg = f"Error processing page {actual_page_num}: {e}"
                ocr_status_queue.put(error_msg)
                rag_status_queue.put(error_msg)
                print(error_msg)
                all_text.append(f"=== Page {actual_page_num} ===\n{error_msg}")
        
        # Clear batch images from memory
        del batch_images
        gc.collect()
        
        # Update progress
        progress = min(100, int((batch_end - start_page + 1) / (end_page - start_page + 1) * 100))
        status_message = f"Progress: {progress}% ({batch_end}/{end_page} pages completed)"
        ocr_status_queue.put(status_message)
        rag_status_queue.put(status_message)
        print(status_message)
    
    # Join all text
    output_text = "\n\n" + "\n".join(all_text) + "\n"
    
    # Save to file
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    
    status_message = f"OCR processing completed! Total pages processed: {len(all_text)}"
    ocr_status_queue.put(status_message)
    rag_status_queue.put(status_message)
    print(status_message)
    
    return output_text

class MultilingualRAG:
    def __init__(self, groq_api_key: str, data_file: str = "output/output.txt"):
        """Initialize the multilingual RAG system"""
        self.groq_api_key = groq_api_key
        self.data_file = data_file
        
        # Initialize multilingual embedding model
        status_message = "Loading multilingual embedding model..."
        rag_status_queue.put(status_message)
        print(status_message)
        
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        
        # Initialize Groq LLM with current production model
        status_message = "Initializing Groq LLM..."
        rag_status_queue.put(status_message)
        print(status_message)
        
        self.llm = ChatGroq(
            temperature=0.7,
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile"  # Current production model
        )
        
        # Initialize components
        self.vector_store = None
        self.documents = []
        self.chunks = []
        
        # Vector store files
        self.vector_store_path = app.config['VECTOR_STORE_FOLDER']
        self.faiss_index_file = f"{self.vector_store_path}/faiss_index.bin"
        self.documents_file = f"{self.vector_store_path}/documents.pkl"
        self.chunks_file = f"{self.vector_store_path}/chunks.pkl"
        
    def detect_language(self, text: str) -> str:
        """Detect the language of input text"""
        # Simple language detection based on script and patterns
        
        # Check for Urdu script (Arabic/Persian characters)
        urdu_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        
        # Check for Roman Urdu patterns (common transliteration words)
        roman_urdu_patterns = [
            r'\b(kya|hai|hain|ka|ke|ki|ko|se|me|per|aur|ya|nahi|han|hum|tum|ap|woh|yeh|kaise|kahan|kab|kyun)\b',
            r'\b(pahar|pani|ghar|sara|sare|wala|wali|wale|ata|ate|ati|karta|karte|karti)\b'
        ]
        
        if urdu_pattern.search(text):
            return "urdu"
        elif any(re.search(pattern, text.lower()) for pattern in roman_urdu_patterns):
            return "roman_urdu"
        else:
            return "english"
    
    def load_and_chunk_documents(self) -> List[Document]:
        """Load and chunk the Urdu text documents"""
        status_message = f"Loading documents from {self.data_file}"
        rag_status_queue.put(status_message)
        print(status_message)
        
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file {self.data_file} not found!")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by page separators
        pages = content.split('=' * 50)
        documents = []
        
        for i, page_content in enumerate(pages):
            if page_content.strip():
                # Clean up the content
                cleaned_content = re.sub(r'=== Page \d+ ===', '', page_content).strip()
                if cleaned_content:
                    documents.append(Document(
                        page_content=cleaned_content,
                        metadata={"page": i + 1, "source": self.data_file}
                    ))
        
        status_message = f"Loaded {len(documents)} pages"
        rag_status_queue.put(status_message)
        print(status_message)
        
        # Chunk the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        status_message = f"Created {len(chunks)} chunks"
        rag_status_queue.put(status_message)
        print(status_message)
        
        self.documents = documents
        self.chunks = chunks
        return chunks
    
    def create_vector_store(self):
        """Create FAISS vector store from documents"""
        status_message = "Creating vector store..."
        rag_status_queue.put(status_message)
        print(status_message)
        
        if not self.chunks:
            self.load_and_chunk_documents()
        
        # Extract text from chunks
        texts = [chunk.page_content for chunk in self.chunks]
        
        # Create embeddings
        status_message = "Creating embeddings..."
        rag_status_queue.put(status_message)
        print(status_message)
        
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        self.vector_store = index
        
        # Save vector store and metadata
        self.save_vector_store()
        
        status_message = f"Vector store created with {index.ntotal} vectors"
        rag_status_queue.put(status_message)
        print(status_message)
    
    def save_vector_store(self):
        """Save vector store and associated data"""
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.vector_store, self.faiss_index_file)
        
        # Save documents and chunks
        with open(self.documents_file, 'wb') as f:
            pickle.dump(self.documents, f)
        
        with open(self.chunks_file, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print("Vector store saved successfully!")
    
    def load_vector_store(self):
        """Load existing vector store"""
        if not os.path.exists(self.faiss_index_file):
            print("No existing vector store found. Creating new one...")
            self.create_vector_store()
            return
        
        print("Loading existing vector store...")
        
        # Load FAISS index
        self.vector_store = faiss.read_index(self.faiss_index_file)
        
        # Load documents and chunks
        with open(self.documents_file, 'rb') as f:
            self.documents = pickle.load(f)
        
        with open(self.chunks_file, 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"Vector store loaded with {self.vector_store.ntotal} vectors")
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[str]:
        """Search for similar chunks using the query"""
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search in vector store
        scores, indices = self.vector_store.search(query_embedding.astype('float32'), k)
        
        # Get relevant chunks
        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                relevant_chunks.append(self.chunks[idx].page_content)
        
        return relevant_chunks
    
    def preprocess_context_for_language(self, context: str, target_language: str) -> str:
        """Preprocess context based on target language to maintain consistency"""
        if target_language == "roman_urdu":
            # For Roman Urdu responses, add instruction to convert any Urdu script
            context_instruction = (
                "Note: The following context contains Urdu text. "
                "Please convert any Urdu script words to Roman Urdu in your response.\n\n"
                f"{context}"
            )
            return context_instruction
        elif target_language == "english":
            # For English responses, add instruction to translate Urdu words
            context_instruction = (
                "Note: The following context contains Urdu text. "
                "Please translate any Urdu words to English in your response and provide English names/terms.\n\n"
                f"{context}"
            )
            return context_instruction
        return context
    
    def post_process_response(self, response: str, language: str) -> str:
        """Post-process response to ensure language consistency"""
        if language == "roman_urdu":
            # Remove any remaining Urdu script characters if they slip through
            urdu_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
            if urdu_pattern.search(response):
                # If Urdu script found, add a note
                response += "\n\n(Note: Some original text was in Urdu script and may need manual conversion to Roman Urdu)"
        elif language == "english":
            # For English responses, check if Urdu script still exists
            urdu_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
            if urdu_pattern.search(response):
                response += "\n\n(Note: Some original names were in Urdu script - please refer to the English transliteration above)"
        
        return response
    
    def create_multilingual_prompt(self, language: str) -> ChatPromptTemplate:
        """Create language-specific prompt templates"""
        
        if language == "urdu":
            template = """آپ ایک مددگار اردو اسسٹنٹ ہیں۔ دیے گئے سیاق و سباق کی بنیاد پر سوال کا جواب دیں۔
جواب اردو میں دیں اور مکمل اور درست ہو۔
سیاق و سباق:
{context}
سوال: {question}
جواب:"""
        
        elif language == "roman_urdu":
            template = """Aap ek helpful Roman Urdu assistant hain. Diye gaye context ki base par sawal ka jawab dein.
Jawab SIRF Roman Urdu mein dein. Urdu script (Arabic letters) ka istemaal BILKUL nahi karein.
Sab kuch English/Roman letters mein likhein. Complete aur sahi jawab dein.
IMPORTANT: Koi bhi Urdu script words ko Roman Urdu mein convert kar dein.
Context:
{context}
Sawal: {question}
Roman Urdu mein jawab:"""
        
        else:  # English
            template = """You are a helpful assistant. Answer the question based on the given context.
Provide a complete and accurate answer in English. If the context contains Urdu names or words, 
please provide their English transliteration (Roman script) or translation.
Context:
{context}
Question: {question}
Answer (in English only):"""
        
        return ChatPromptTemplate.from_template(template)
    
    def answer_query(self, query: str) -> str:
        """Answer a multilingual query"""
        # Detect language
        language = self.detect_language(query)
        print(f"Detected language: {language}")
        
        # Search for relevant chunks
        relevant_chunks = self.search_similar_chunks(query, k=5)
        context = "\n\n".join(relevant_chunks)
        
        # Preprocess context for target language
        context = self.preprocess_context_for_language(context, language)
        
        # Create appropriate prompt
        prompt_template = self.create_multilingual_prompt(language)
        
        # Create and run chain using new LangChain syntax
        try:
            # Use the new RunnableSequence approach
            chain = prompt_template | self.llm
            response = chain.invoke({"context": context, "question": query})
            
            # Extract content from the response
            if hasattr(response, 'content'):
                final_response = response.content.strip()
            else:
                final_response = str(response).strip()
            
            # Post-process for language consistency
            final_response = self.post_process_response(final_response, language)
            return final_response
                
        except Exception as e:
            return f"Error generating response: {str(e)}"

@app.route('/')
def index():
    """Render the main page"""
    return render_template('frontend.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Handle PDF upload"""
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['pdf_file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get page range
        start_page = int(request.form.get('start_page', 1))
        end_page = int(request.form.get('end_page', 10))
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'start_page': start_page,
            'end_page': end_page
        })
    
    return jsonify({'error': 'Invalid file format. Please upload a PDF file.'}), 400

@app.route('/apply_ocr', methods=['POST'])
def apply_ocr():
    """Apply OCR to the uploaded PDF"""
    # Clear status queues before starting new operation
    clear_status_queues()
    
    data = request.json
    filepath = data.get('filepath')
    start_page = data.get('start_page', 1)
    end_page = data.get('end_page', 10)
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 400
    
    try:
        # Use the new batch processing function
        output_text = process_pdf_batch(filepath, start_page, end_page)
        
        if output_text is None:
            return jsonify({'error': 'Failed to process PDF'}), 500
        
        return jsonify({
            'success': True,
            'text': output_text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/configure_rag', methods=['POST'])
def configure_rag():
    """Configure the RAG system"""
    # Clear status queues before starting new operation
    clear_status_queues()
    
    data = request.json
    filepath = data.get('filepath')
    start_page = data.get('start_page', 1)
    end_page = data.get('end_page', 10)
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 400
    
    try:
        # Use the new batch processing function
        output_text = process_pdf_batch(filepath, start_page, end_page)
        
        if output_text is None:
            return jsonify({'error': 'Failed to process PDF'}), 500
        
        # Initialize RAG system
        global rag_system
        load_dotenv()
        groq_api_key = os.getenv('GROQ_API_KEY')
        
        if not groq_api_key:
            return jsonify({'error': 'Groq API key not found'}), 500
        
        rag_system = MultilingualRAG(groq_api_key)
        
        # Create vector store
        rag_system.create_vector_store()
        
        return jsonify({
            'success': True,
            'message': 'RAG system configured successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat queries"""
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    global rag_system
    if rag_system is None:
        return jsonify({'error': 'RAG system not configured'}), 400
    
    try:
        response = rag_system.answer_query(query)
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_ocr_status', methods=['GET'])
def get_ocr_status():
    """Get OCR status updates"""
    status_messages = []
    while not ocr_status_queue.empty():
        status_messages.append(ocr_status_queue.get())
    
    return jsonify({
        'status_messages': status_messages,
        'has_updates': len(status_messages) > 0
    })

@app.route('/get_rag_status', methods=['GET'])
def get_rag_status():
    """Get RAG status updates"""
    status_messages = []
    while not rag_status_queue.empty():
        status_messages.append(rag_status_queue.get())
    
    return jsonify({
        'status_messages': status_messages,
        'has_updates': len(status_messages) > 0
    })

@app.route('/get_combined_status', methods=['GET'])
def get_combined_status():
    """Get both OCR and RAG status updates for the RAG configuration process"""
    ocr_messages = []
    rag_messages = []
    
    while not ocr_status_queue.empty():
        ocr_messages.append(ocr_status_queue.get())
    
    while not rag_status_queue.empty():
        rag_messages.append(rag_status_queue.get())
    
    return jsonify({
        'ocr_status_messages': ocr_messages,
        'rag_status_messages': rag_messages,
        'has_updates': len(ocr_messages) > 0 or len(rag_messages) > 0
    })

if __name__ == '__main__':
    # Load models on startup
    setup_models()
    app.run(host="0.0.0.0", port=5005, debug=True)