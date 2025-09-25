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
import json

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
book_library = {}  # Store multiple books and their RAG systems
current_selected_book = None
book_library_file = "book_library.json"

# Global embedding model instance (shared)
global_embedding_model = None
global_embedding_model_lock = threading.Lock()

# Status queues for real-time updates
ocr_status_queue = Queue()
rag_status_queue = Queue()

def get_shared_embedding_model():
    """Get or create shared embedding model instance"""
    global global_embedding_model

    if global_embedding_model is None:
        with global_embedding_model_lock:
            # Double-check locking pattern
            if global_embedding_model is None:
                print("Loading shared multilingual embedding model...")
                status_message = "Loading multilingual embedding model (shared)..."
                rag_status_queue.put(status_message)

                global_embedding_model = SentenceTransformer(
                    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
                )
                print("Shared embedding model loaded successfully!")

    return global_embedding_model

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
    recognition_model.load_state_dict(torch.load("best_norm_ED.pth", map_location=device, weights_only=True))
    recognition_model.eval()

    detection_model = YOLO("yolov8m_UrduDoc.pt")

    print("OCR models loaded successfully!")

    # Pre-load the shared embedding model
    print("Pre-loading shared embedding model...")
    get_shared_embedding_model()
    print("All models loaded successfully!")

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
    def __init__(self, groq_api_key: str, data_file: str = "output/output.txt", book_id: str = None):
        """Initialize the multilingual RAG system"""
        self.groq_api_key = groq_api_key
        self.data_file = data_file
        self.book_id = book_id or os.path.splitext(os.path.basename(data_file))[0]

        # Use shared embedding model instead of creating new one
        status_message = "Getting shared embedding model..."
        rag_status_queue.put(status_message)
        print(status_message)

        self.embedding_model = get_shared_embedding_model()

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

        # Vector store files - unique per book
        self.vector_store_path = os.path.join(app.config['VECTOR_STORE_FOLDER'], self.book_id)
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

        print(f"Vector store saved successfully for book: {self.book_id}!")

    def load_vector_store(self):
        """Load existing vector store"""
        try:
            if not os.path.exists(self.faiss_index_file):
                print(f"No existing vector store found for book: {self.book_id}. Creating new one...")
                self.create_vector_store()
                return

            print(f"Loading existing vector store for book: {self.book_id}...")

            # Check if all required files exist
            if not os.path.exists(self.documents_file):
                print(f"Documents file missing for book: {self.book_id}")
                self.create_vector_store()
                return

            if not os.path.exists(self.chunks_file):
                print(f"Chunks file missing for book: {self.book_id}")
                self.create_vector_store()
                return

            # Load FAISS index
            try:
                self.vector_store = faiss.read_index(self.faiss_index_file)
            except Exception as faiss_error:
                print(f"Error loading FAISS index for book {self.book_id}: {faiss_error}")
                print("Recreating vector store...")
                self.create_vector_store()
                return

            # Load documents and chunks
            try:
                with open(self.documents_file, 'rb') as f:
                    self.documents = pickle.load(f)

                with open(self.chunks_file, 'rb') as f:
                    self.chunks = pickle.load(f)

                print(f"Vector store loaded with {self.vector_store.ntotal} vectors for book: {self.book_id}")
            except Exception as pickle_error:
                print(f"Error loading pickle files for book {self.book_id}: {pickle_error}")
                print("Recreating vector store...")
                self.create_vector_store()
                return

        except Exception as e:
            print(f"Unexpected error in load_vector_store for book {self.book_id}: {e}")
            print("Recreating vector store...")
            self.create_vector_store()
    
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
    return render_template('frontend1.html')

@app.route('/upload_and_ingest', methods=['POST'])
def upload_and_ingest():
    """Handle PDF upload and RAG configuration in one endpoint"""
    # Clear status queues before starting new operation
    clear_status_queues()

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

        try:
            # Get book filename for unique ID
            book_filename = os.path.basename(filepath)
            book_name = os.path.splitext(book_filename)[0]

            # Check if book with same name already exists
            if check_duplicate_book(book_name):
                return jsonify({
                    'error': f'A book named "{book_name}" already exists in the library. Please rename your file and try again.'
                }), 400

            book_id = book_name  # Use filename as book ID

            # Use the new batch processing function
            output_text = process_pdf_batch(filepath, start_page, end_page)

            if output_text is None:
                return jsonify({'error': 'Failed to process PDF'}), 500

            # Create output file with book-specific name
            book_output_file = f"output/{book_id}.txt"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{book_id}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output_text)

            # Initialize RAG system with book-specific ID
            global book_library
            load_dotenv()
            groq_api_key = os.getenv('GROQ_API_KEY')

            if not groq_api_key:
                return jsonify({'error': 'Groq API key not found'}), 500

            # Create RAG system for this specific book
            book_rag_system = MultilingualRAG(groq_api_key, book_output_file, book_id)

            # Create vector store for this book
            book_rag_system.create_vector_store()

            # Add to book library
            book_library[book_id] = {
                'name': book_name,
                'filename': book_filename,
                'rag_system': book_rag_system,
                'output_file': book_output_file,
                'created_at': os.path.getctime(filepath)
            }

            # Save book library to file
            save_book_library()

            return jsonify({
                'success': True,
                'message': f'Book successfully uploaded and ingested: {book_name}',
                'book_id': book_id,
                'book_name': book_name,
                'total_books': len(book_library)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file format. Please upload a PDF file.'}), 400


@app.route('/delete_book', methods=['POST'])
def delete_book():
    """Delete a book from the library and its vector database"""
    try:
        data = request.json
        book_id = data.get('book_id')

        if not book_id:
            return jsonify({'error': 'No book ID provided'}), 400

        global book_library, current_selected_book

        if book_id not in book_library:
            return jsonify({'error': 'Book not found in library'}), 404

        book_info = book_library[book_id]
        book_name = book_info['name']

        print(f"Deleting book: {book_id} - {book_name}")

        # Remove the book from library
        del book_library[book_id]

        # Clear current selection if this book was selected
        if current_selected_book == book_id:
            current_selected_book = None

        # Remove vector store files
        vector_store_path = os.path.join(app.config['VECTOR_STORE_FOLDER'], book_id)
        if os.path.exists(vector_store_path):
            print(f"Removing vector store directory: {vector_store_path}")
            import shutil
            shutil.rmtree(vector_store_path)
            print("Vector store directory removed successfully")

        # Remove output text file
        output_file = book_info['output_file']
        if os.path.exists(output_file):
            print(f"Removing output file: {output_file}")
            os.remove(output_file)
            print("Output file removed successfully")

        # Save updated book library
        save_book_library()
        print(f"Book library saved. Remaining books: {len(book_library)}")

        return jsonify({
            'success': True,
            'message': f'Book "{book_name}" deleted successfully',
            'total_books': len(book_library)
        })

    except Exception as e:
        print(f"Error deleting book: {e}")
        return jsonify({'error': f'Error deleting book: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat queries with selected book"""
    data = request.json
    query = data.get('query')
    selected_book_id = data.get('book_id')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    global book_library, current_selected_book
    if not book_library:
        return jsonify({'error': 'No books in library. Please configure RAG first.'}), 400

    # Use selected book or current selected book
    book_id = selected_book_id or current_selected_book
    if not book_id or book_id not in book_library:
        return jsonify({'error': 'No book selected or invalid book ID'}), 400

    try:
        rag_system = book_library[book_id]['rag_system']
        response = rag_system.answer_query(query)
        return jsonify({
            'success': True,
            'response': response,
            'book_name': book_library[book_id]['name']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_books', methods=['GET'])
def get_books():
    """Get list of all books in the library"""
    global book_library
    books = []
    for book_id, book_info in book_library.items():
        books.append({
            'id': book_id,
            'name': book_info['name'],
            'filename': book_info['filename'],
            'created_at': book_info['created_at']
        })

    # Sort by creation time (newest first)
    books.sort(key=lambda x: x['created_at'], reverse=True)

    return jsonify({
        'success': True,
        'books': books,
        'total_books': len(books)
    })

@app.route('/select_book', methods=['POST'])
def select_book():
    """Select a book for chatting"""
    data = request.json
    book_id = data.get('book_id')

    global book_library, current_selected_book
    if not book_id or book_id not in book_library:
        return jsonify({'error': 'Invalid book ID'}), 400

    current_selected_book = book_id

    return jsonify({
        'success': True,
        'message': f'Selected book: {book_library[book_id]["name"]}',
        'book_name': book_library[book_id]['name']
    })

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

def save_book_library():
    """Save book library to JSON file"""
    try:
        library_data = {}
        for book_id, book_info in book_library.items():
            # Convert RAG system to serializable format (exclude non-serializable objects)
            library_data[book_id] = {
                'name': book_info['name'],
                'filename': book_info['filename'],
                'output_file': book_info['output_file'],
                'created_at': book_info['created_at']
            }

        with open(book_library_file, 'w', encoding='utf-8') as f:
            json.dump(library_data, f, indent=2)
    except Exception as e:
        print(f"Error saving book library: {e}")

def load_book_library():
    """Load book library from JSON file"""
    global book_library
    try:
        if os.path.exists(book_library_file):
            with open(book_library_file, 'r', encoding='utf-8') as f:
                library_data = json.load(f)

            print(f"Loading {len(library_data)} books from library...")

            # Pre-load shared embedding model before loading books
            print("Pre-loading shared embedding model for existing books...")
            get_shared_embedding_model()

            # Reinitialize RAG systems for loaded books
            load_dotenv()
            groq_api_key = os.getenv('GROQ_API_KEY')

            if not groq_api_key:
                print("Warning: GROQ_API_KEY not found, books will be loaded without RAG systems")
                return

            for book_id, book_info in library_data.items():
                try:
                    # Create output file path if it doesn't exist
                    output_file = book_info.get('output_file', f"output/{book_id}.txt")

                    # Create RAG system for this book
                    print(f"Initializing RAG system for book: {book_id}")
                    book_rag_system = MultilingualRAG(groq_api_key, output_file, book_id)

                    # Load existing vector store if it exists
                    if os.path.exists(book_rag_system.faiss_index_file):
                        print(f"Loading vector store for book: {book_id}")
                        book_rag_system.load_vector_store()
                        print(f"Successfully loaded vector store for book: {book_id}")
                    else:
                        print(f"Vector store not found for book {book_id}, will be created when needed")

                    # Add to book library
                    book_library[book_id] = {
                        'name': book_info['name'],
                        'filename': book_info['filename'],
                        'rag_system': book_rag_system,
                        'output_file': output_file,
                        'created_at': book_info.get('created_at', 0)
                    }
                    print(f"Successfully loaded book: {book_id}")
                except Exception as e:
                    print(f"Error loading book {book_id}: {e}")
                    # Continue loading other books even if one fails
                    continue
        else:
            print("No existing book library file found, starting fresh")
    except Exception as e:
        print(f"Error loading book library: {e}")

def check_duplicate_book(book_name):
    """Check if a book with the same name already exists"""
    for book_id, book_info in book_library.items():
        if book_info['name'] == book_name:
            return True
    return False

if __name__ == '__main__':
    # Load book library on startup
    load_book_library()
    # Load models on startup
    setup_models()
    app.run(host="0.0.0.0", port=5001, debug=True)