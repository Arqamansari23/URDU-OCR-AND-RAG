# import os
# import sys
# from pathlib import Path
# from typing import List, Optional
# import faiss
# import numpy as np
# import pickle
# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate
# import re
# import argparse
# from dotenv import load_dotenv

# class MultilingualRAG:
#     def __init__(self, groq_api_key: str, data_file: str = "output/output.txt"):
#         """Initialize the multilingual RAG system"""
#         self.groq_api_key = groq_api_key
#         self.data_file = data_file
        
#         # Initialize multilingual embedding model
#         print("Loading multilingual embedding model...")
#         self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        
#         # Initialize Groq LLM with current production model
#         print("Initializing Groq LLM...")
#         self.llm = ChatGroq(
#             temperature=0.7,
#             groq_api_key=groq_api_key,
#             model_name="llama-3.3-70b-versatile"  # Current production model
#         )
        
#         # Initialize components
#         self.vector_store = None
#         self.documents = []
#         self.chunks = []
        
#         # Vector store files
#         self.vector_store_path = "vector_store"
#         self.faiss_index_file = f"{self.vector_store_path}/faiss_index.bin"
#         self.documents_file = f"{self.vector_store_path}/documents.pkl"
#         self.chunks_file = f"{self.vector_store_path}/chunks.pkl"
        
#     def detect_language(self, text: str) -> str:
#         """Detect the language of input text"""
#         # Simple language detection based on script and patterns
        
#         # Check for Urdu script (Arabic/Persian characters)
#         urdu_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        
#         # Check for Roman Urdu patterns (common transliteration words)
#         roman_urdu_patterns = [
#             r'\b(kya|hai|hain|ka|ke|ki|ko|se|me|per|aur|ya|nahi|han|hum|tum|ap|woh|yeh|kaise|kahan|kab|kyun)\b',
#             r'\b(pahar|pani|ghar|sara|sare|wala|wali|wale|ata|ate|ati|karta|karte|karti)\b'
#         ]
        
#         if urdu_pattern.search(text):
#             return "urdu"
#         elif any(re.search(pattern, text.lower()) for pattern in roman_urdu_patterns):
#             return "roman_urdu"
#         else:
#             return "english"
    
#     def load_and_chunk_documents(self) -> List[Document]:
#         """Load and chunk the Urdu text documents"""
#         print(f"Loading documents from {self.data_file}")
        
#         if not os.path.exists(self.data_file):
#             raise FileNotFoundError(f"Data file {self.data_file} not found!")
        
#         with open(self.data_file, 'r', encoding='utf-8') as f:
#             content = f.read()
        
#         # Split by page separators
#         pages = content.split('=' * 50)
#         documents = []
        
#         for i, page_content in enumerate(pages):
#             if page_content.strip():
#                 # Clean up the content
#                 cleaned_content = re.sub(r'=== Page \d+ ===', '', page_content).strip()
#                 if cleaned_content:
#                     documents.append(Document(
#                         page_content=cleaned_content,
#                         metadata={"page": i + 1, "source": self.data_file}
#                     ))
        
#         print(f"Loaded {len(documents)} pages")
        
#         # Chunk the documents
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=500,
#             chunk_overlap=100,
#             separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
#         )
        
#         chunks = text_splitter.split_documents(documents)
#         print(f"Created {len(chunks)} chunks")
        
#         self.documents = documents
#         self.chunks = chunks
#         return chunks
    
#     def create_vector_store(self):
#         """Create FAISS vector store from documents"""
#         print("Creating vector store...")
        
#         if not self.chunks:
#             self.load_and_chunk_documents()
        
#         # Extract text from chunks
#         texts = [chunk.page_content for chunk in self.chunks]
        
#         # Create embeddings
#         print("Creating embeddings...")
#         embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
#         # Create FAISS index
#         dimension = embeddings.shape[1]
#         index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
#         # Normalize embeddings for cosine similarity
#         faiss.normalize_L2(embeddings)
#         index.add(embeddings.astype('float32'))
        
#         self.vector_store = index
        
#         # Save vector store and metadata
#         self.save_vector_store()
        
#         print(f"Vector store created with {index.ntotal} vectors")
    
#     def save_vector_store(self):
#         """Save vector store and associated data"""
#         os.makedirs(self.vector_store_path, exist_ok=True)
        
#         # Save FAISS index
#         faiss.write_index(self.vector_store, self.faiss_index_file)
        
#         # Save documents and chunks
#         with open(self.documents_file, 'wb') as f:
#             pickle.dump(self.documents, f)
        
#         with open(self.chunks_file, 'wb') as f:
#             pickle.dump(self.chunks, f)
        
#         print("Vector store saved successfully!")
    
#     def load_vector_store(self):
#         """Load existing vector store"""
#         if not os.path.exists(self.faiss_index_file):
#             print("No existing vector store found. Creating new one...")
#             self.create_vector_store()
#             return
        
#         print("Loading existing vector store...")
        
#         # Load FAISS index
#         self.vector_store = faiss.read_index(self.faiss_index_file)
        
#         # Load documents and chunks
#         with open(self.documents_file, 'rb') as f:
#             self.documents = pickle.load(f)
        
#         with open(self.chunks_file, 'rb') as f:
#             self.chunks = pickle.load(f)
        
#         print(f"Vector store loaded with {self.vector_store.ntotal} vectors")
    
#     def search_similar_chunks(self, query: str, k: int = 5) -> List[str]:
#         """Search for similar chunks using the query"""
#         # Create query embedding
#         query_embedding = self.embedding_model.encode([query])
#         faiss.normalize_L2(query_embedding)
        
#         # Search in vector store
#         scores, indices = self.vector_store.search(query_embedding.astype('float32'), k)
        
#         # Get relevant chunks
#         relevant_chunks = []
#         for idx in indices[0]:
#             if idx < len(self.chunks):
#                 relevant_chunks.append(self.chunks[idx].page_content)
        
#         return relevant_chunks
    
#     def preprocess_context_for_language(self, context: str, target_language: str) -> str:
#         """Preprocess context based on target language to maintain consistency"""
#         if target_language == "roman_urdu":
#             # For Roman Urdu responses, add instruction to convert any Urdu script
#             context_instruction = (
#                 "Note: The following context contains Urdu text. "
#                 "Please convert any Urdu script words to Roman Urdu in your response.\n\n"
#                 f"{context}"
#             )
#             return context_instruction
#         elif target_language == "english":
#             # For English responses, add instruction to translate Urdu words
#             context_instruction = (
#                 "Note: The following context contains Urdu text. "
#                 "Please translate any Urdu words to English in your response and provide English names/terms.\n\n"
#                 f"{context}"
#             )
#             return context_instruction
#         return context
    
#     def post_process_response(self, response: str, language: str) -> str:
#         """Post-process response to ensure language consistency"""
#         if language == "roman_urdu":
#             # Remove any remaining Urdu script characters if they slip through
#             urdu_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
#             if urdu_pattern.search(response):
#                 # If Urdu script found, add a note
#                 response += "\n\n(Note: Some original text was in Urdu script and may need manual conversion to Roman Urdu)"
#         elif language == "english":
#             # For English responses, check if Urdu script still exists
#             urdu_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
#             if urdu_pattern.search(response):
#                 response += "\n\n(Note: Some original names were in Urdu script - please refer to the English transliteration above)"
        
#         return response
    
#     def create_multilingual_prompt(self, language: str) -> ChatPromptTemplate:
#         """Create language-specific prompt templates"""
        
#         if language == "urdu":
#             template = """آپ ایک مددگار اردو اسسٹنٹ ہیں۔ دیے گئے سیاق و سباق کی بنیاد پر سوال کا جواب دیں۔
# جواب اردو میں دیں اور مکمل اور درست ہو۔

# سیاق و سباق:
# {context}

# سوال: {question}

# جواب:"""
        
#         elif language == "roman_urdu":
#             template = """Aap ek helpful Roman Urdu assistant hain. Diye gaye context ki base par sawal ka jawab dein.
# Jawab SIRF Roman Urdu mein dein. Urdu script (Arabic letters) ka istemaal BILKUL nahi karein.
# Sab kuch English/Roman letters mein likhein. Complete aur sahi jawab dein.

# IMPORTANT: Koi bhi Urdu script words ko Roman Urdu mein convert kar dein.

# Context:
# {context}

# Sawal: {question}

# Roman Urdu mein jawab:"""
        
#         else:  # English
#             template = """You are a helpful assistant. Answer the question based on the given context.
# Provide a complete and accurate answer in English. If the context contains Urdu names or words, 
# please provide their English transliteration (Roman script) or translation.

# Context:
# {context}

# Question: {question}

# Answer (in English only):"""
        
#         return ChatPromptTemplate.from_template(template)
    
#     def answer_query(self, query: str) -> str:
#         """Answer a multilingual query"""
#         # Detect language
#         language = self.detect_language(query)
#         print(f"Detected language: {language}")
        
#         # Search for relevant chunks
#         relevant_chunks = self.search_similar_chunks(query, k=5)
#         context = "\n\n".join(relevant_chunks)
        
#         # Preprocess context for target language
#         context = self.preprocess_context_for_language(context, language)
        
#         # Create appropriate prompt
#         prompt_template = self.create_multilingual_prompt(language)
        
#         # Create and run chain using new LangChain syntax
#         try:
#             # Use the new RunnableSequence approach
#             chain = prompt_template | self.llm
#             response = chain.invoke({"context": context, "question": query})
            
#             # Extract content from the response
#             if hasattr(response, 'content'):
#                 final_response = response.content.strip()
#             else:
#                 final_response = str(response).strip()
            
#             # Post-process for language consistency
#             final_response = self.post_process_response(final_response, language)
#             return final_response
                
#         except Exception as e:
#             return f"Error generating response: {str(e)}"
    
#     def interactive_chat(self):
#         """Interactive chat interface"""
#         print("\n" + "="*60)
#         print("🤖 Multilingual Urdu RAG Chat System")
#         print("="*60)
#         print("Languages supported:")
#         print("• English: How did Romans come to the hills?")
#         print("• اردو: رومی پہاڑوں پر کیسے آئے؟")
#         print("• Roman Urdu: Romans pahar par kaise aye?")
#         print("\nType 'quit', 'exit', or 'q' to end the conversation")
#         print("-" * 60)
        
#         while True:
#             try:
#                 # Get user input
#                 query = input("\n👤 You: ").strip()
                
#                 # Check for exit commands
#                 if query.lower() in ['quit', 'exit', 'q', 'خروج']:
#                     print("\n👋 Goodbye! خدا حافظ! Allah hafiz!")
#                     break
                
#                 if not query:
#                     continue
                
#                 # Show loading message
#                 print("🔍 Searching and generating response...")
                
#                 # Get answer
#                 answer = self.answer_query(query)
                
#                 # Display answer
#                 print(f"\n🤖 Bot: {answer}")
#                 print("-" * 60)
                
#             except KeyboardInterrupt:
#                 print("\n\n👋 Chat ended by user. Goodbye!")
#                 break
#             except Exception as e:
#                 print(f"\n❌ Error: {e}")

# def main():
#     """Main function"""
#     # Load environment variables from .env file
#     load_dotenv()
    
#     parser = argparse.ArgumentParser(description='Multilingual Urdu RAG Application')
#     parser.add_argument('--data', default='output/output.txt', help='Path to the OCR text file')
#     parser.add_argument('--groq-key', help='Groq API key (overrides .env file and environment variable)')
#     parser.add_argument('--rebuild-index', action='store_true', help='Force rebuild of vector index')
    
#     args = parser.parse_args()
    
#     # Get Groq API key with priority: command line > .env file > environment variable
#     groq_api_key = args.groq_key or os.getenv('GROQ_API_KEY')
    
#     if not groq_api_key:
#         print("❌ Error: Groq API key not found!")
#         print("Please provide your Groq API key using one of these methods:")
#         print("1. Create a .env file with: GROQ_API_KEY=your_api_key_here")
#         print("2. Set environment variable: export GROQ_API_KEY=your_api_key_here")
#         print("3. Use command line: --groq-key your_api_key_here")
#         print("\nGet your API key from: https://console.groq.com/")
#         sys.exit(1)
    
#     try:
#         # Initialize RAG system
#         print("🚀 Initializing Multilingual Urdu RAG System...")
#         rag_system = MultilingualRAG(groq_api_key, args.data)
        
#         # Load or create vector store
#         if args.rebuild_index or not os.path.exists(rag_system.faiss_index_file):
#             rag_system.create_vector_store()
#         else:
#             rag_system.load_vector_store()
        
#         # Start interactive chat
#         rag_system.interactive_chat()
        
#     except FileNotFoundError as e:
#         print(f"❌ Error: {e}")
#         print("Please make sure the OCR output file exists at the specified path.")
#     except Exception as e:
#         print(f"❌ Unexpected error: {e}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()





# import os
# import sys
# from pathlib import Path
# from typing import List, Optional
# import faiss
# import numpy as np
# import pickle
# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate
# import re
# import argparse
# from dotenv import load_dotenv

# class MultilingualRAG:
#     def __init__(self, groq_api_key: str, data_file: str = "output/output.txt", vector_store_path: str = "vector_store"):
#         """Initialize the multilingual RAG system"""
#         self.groq_api_key = groq_api_key
#         self.data_file = data_file
#         self.vector_store_path = vector_store_path
        
#         # Initialize multilingual embedding model
#         print("Loading multilingual embedding model...")
#         self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        
#         # Initialize Groq LLM with current production model
#         print("Initializing Groq LLM...")
#         self.llm = ChatGroq(
#             temperature=0.7,
#             groq_api_key=groq_api_key,
#             model_name="llama-3.3-70b-versatile"  # Current production model
#         )
        
#         # Initialize components
#         self.vector_store = None
#         self.documents = []
#         self.chunks = []
        
#         # Vector store files
#         self.faiss_index_file = f"{self.vector_store_path}/faiss_index.bin"
#         self.documents_file = f"{self.vector_store_path}/documents.pkl"
#         self.chunks_file = f"{self.vector_store_path}/chunks.pkl"
        
#     def detect_language(self, text: str) -> str:
#         """Detect the language of input text"""
#         # Simple language detection based on script and patterns
        
#         # Check for Urdu script (Arabic/Persian characters)
#         urdu_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        
#         # Check for Roman Urdu patterns (common transliteration words)
#         roman_urdu_patterns = [
#             r'\b(kya|hai|hain|ka|ke|ki|ko|se|me|per|aur|ya|nahi|han|hum|tum|ap|woh|yeh|kaise|kahan|kab|kyun)\b',
#             r'\b(pahar|pani|ghar|sara|sare|wala|wali|wale|ata|ate|ati|karta|karte|karti)\b'
#         ]
        
#         if urdu_pattern.search(text):
#             return "urdu"
#         elif any(re.search(pattern, text.lower()) for pattern in roman_urdu_patterns):
#             return "roman_urdu"
#         else:
#             return "english"
    
#     def load_and_chunk_documents(self) -> List[Document]:
#         """Load and chunk the Urdu text documents"""
#         print(f"Loading documents from {self.data_file}")
        
#         if not os.path.exists(self.data_file):
#             raise FileNotFoundError(f"Data file {self.data_file} not found!")
        
#         with open(self.data_file, 'r', encoding='utf-8') as f:
#             content = f.read()
        
#         # Split by page separators
#         pages = content.split('=' * 50)
#         documents = []
        
#         for i, page_content in enumerate(pages):
#             if page_content.strip():
#                 # Clean up the content
#                 cleaned_content = re.sub(r'=== Page \d+ ===', '', page_content).strip()
#                 if cleaned_content:
#                     documents.append(Document(
#                         page_content=cleaned_content,
#                         metadata={"page": i + 1, "source": self.data_file}
#                     ))
        
#         print(f"Loaded {len(documents)} pages")
        
#         # Chunk the documents
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=500,
#             chunk_overlap=100,
#             separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
#         )
        
#         chunks = text_splitter.split_documents(documents)
#         print(f"Created {len(chunks)} chunks")
        
#         self.documents = documents
#         self.chunks = chunks
#         return chunks
    
#     def create_vector_store(self):
#         """Create FAISS vector store from documents"""
#         print("Creating vector store...")
        
#         if not self.chunks:
#             self.load_and_chunk_documents()
        
#         # Extract text from chunks
#         texts = [chunk.page_content for chunk in self.chunks]
        
#         # Create embeddings
#         print("Creating embeddings...")
#         embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
#         # Create FAISS index
#         dimension = embeddings.shape[1]
#         index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
#         # Normalize embeddings for cosine similarity
#         faiss.normalize_L2(embeddings)
#         index.add(embeddings.astype('float32'))
        
#         self.vector_store = index
        
#         # Save vector store and metadata
#         self.save_vector_store()
        
#         print(f"Vector store created with {index.ntotal} vectors")
    
#     def save_vector_store(self):
#         """Save vector store and associated data"""
#         os.makedirs(self.vector_store_path, exist_ok=True)
        
#         # Save FAISS index
#         faiss.write_index(self.vector_store, self.faiss_index_file)
        
#         # Save documents and chunks
#         with open(self.documents_file, 'wb') as f:
#             pickle.dump(self.documents, f)
        
#         with open(self.chunks_file, 'wb') as f:
#             pickle.dump(self.chunks, f)
        
#         print("Vector store saved successfully!")
    
#     def load_vector_store(self):
#         """Load existing vector store"""
#         if not os.path.exists(self.faiss_index_file):
#             print("No existing vector store found. Creating new one...")
#             self.create_vector_store()
#             return
        
#         print("Loading existing vector store...")
        
#         # Load FAISS index
#         self.vector_store = faiss.read_index(self.faiss_index_file)
        
#         # Load documents and chunks
#         with open(self.documents_file, 'rb') as f:
#             self.documents = pickle.load(f)
        
#         with open(self.chunks_file, 'rb') as f:
#             self.chunks = pickle.load(f)
        
#         print(f"Vector store loaded with {self.vector_store.ntotal} vectors")
    
#     def search_similar_chunks(self, query: str, k: int = 5) -> List[str]:
#         """Search for similar chunks using the query"""
#         # Create query embedding
#         query_embedding = self.embedding_model.encode([query])  # Fixed: Added missing brackets
#         faiss.normalize_L2(query_embedding)
        
#         # Search in vector store
#         scores, indices = self.vector_store.search(query_embedding.astype('float32'), k)
        
#         # Get relevant chunks
#         relevant_chunks = []
#         for idx in indices[0]:
#             if idx < len(self.chunks):
#                 relevant_chunks.append(self.chunks[idx].page_content)
        
#         return relevant_chunks
    
#     def preprocess_context_for_language(self, context: str, target_language: str) -> str:
#         """Preprocess context based on target language to maintain consistency"""
#         if target_language == "roman_urdu":
#             # For Roman Urdu responses, add instruction to convert any Urdu script
#             context_instruction = (
#                 "Note: The following context contains Urdu text. "
#                 "Please convert any Urdu script words to Roman Urdu in your response.\n\n"
#                 f"{context}"
#             )
#             return context_instruction
#         elif target_language == "english":
#             # For English responses, add instruction to translate Urdu words
#             context_instruction = (
#                 "Note: The following context contains Urdu text. "
#                 "Please translate any Urdu words to English in your response and provide English names/terms.\n\n"
#                 f"{context}"
#             )
#             return context_instruction
#         return context
    
#     def post_process_response(self, response: str, language: str) -> str:
#         """Post-process response to ensure language consistency"""
#         if language == "roman_urdu":
#             # Remove any remaining Urdu script characters if they slip through
#             urdu_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
#             if urdu_pattern.search(response):
#                 # If Urdu script found, add a note
#                 response += "\n\n(Note: Some original text was in Urdu script and may need manual conversion to Roman Urdu)"
#         elif language == "english":
#             # For English responses, check if Urdu script still exists
#             urdu_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
#             if urdu_pattern.search(response):
#                 response += "\n\n(Note: Some original names were in Urdu script - please refer to the English transliteration above)"
        
#         return response
    
#     def create_multilingual_prompt(self, language: str) -> ChatPromptTemplate:
#         """Create language-specific prompt templates"""
        
#         if language == "urdu":
#             template = """آپ ایک مددگار اردو اسسٹنٹ ہیں۔ دیے گئے سیاق و سباق کی بنیاد پر سوال کا جواب دیں۔
# جواب اردو میں دیں اور مکمل اور درست ہو۔

# سیاق و سباق:
# {context}

# سوال: {question}

# جواب:"""
        
#         elif language == "roman_urdu":
#             template = """Aap ek helpful Roman Urdu assistant hain. Diye gaye context ki base par sawal ka jawab dein.
# Jawab SIRF Roman Urdu mein dein. Urdu script (Arabic letters) ka istemaal BILKUL nahi karein.
# Sab kuch English/Roman letters mein likhein. Complete aur sahi jawab dein.

# IMPORTANT: Koi bhi Urdu script words ko Roman Urdu mein convert kar dein.

# Context:
# {context}

# Sawal: {question}

# Roman Urdu mein jawab:"""
        
#         else:  # English
#             template = """You are a helpful assistant. Answer the question based on the given context.
# Provide a complete and accurate answer in English. If the context contains Urdu names or words, 
# please provide their English transliteration (Roman script) or translation.

# Context:
# {context}

# Question: {question}

# Answer (in English only):"""
        
#         return ChatPromptTemplate.from_template(template)
    
#     def answer_query(self, query: str) -> str:
#         """Answer a multilingual query"""
#         # Detect language
#         language = self.detect_language(query)
#         print(f"Detected language: {language}")
        
#         # Search for relevant chunks
#         relevant_chunks = self.search_similar_chunks(query, k=5)
#         context = "\n\n".join(relevant_chunks)
        
#         # Preprocess context for target language
#         context = self.preprocess_context_for_language(context, language)
        
#         # Create appropriate prompt
#         prompt_template = self.create_multilingual_prompt(language)
        
#         # Create and run chain using new LangChain syntax
#         try:
#             # Use the new RunnableSequence approach
#             chain = prompt_template | self.llm
#             response = chain.invoke({"context": context, "question": query})
            
#             # Extract content from the response
#             if hasattr(response, 'content'):
#                 final_response = response.content.strip()
#             else:
#                 final_response = str(response).strip()
            
#             # Post-process for language consistency
#             final_response = self.post_process_response(final_response, language)
#             return final_response
                
#         except Exception as e:
#             return f"Error generating response: {str(e)}"

# def main():
#     """Main function"""
#     # Load environment variables from .env file
#     load_dotenv()
    
#     parser = argparse.ArgumentParser(description='Multilingual Urdu RAG Application')
#     parser.add_argument('--data', default='output/output.txt', help='Path to the OCR text file')
#     parser.add_argument('--groq-key', help='Groq API key (overrides .env file and environment variable)')
#     parser.add_argument('--rebuild-index', action='store_true', help='Force rebuild of vector index')
    
#     args = parser.parse_args()
    
#     # Get Groq API key with priority: command line > .env file > environment variable
#     groq_api_key = args.groq_key or os.getenv('GROQ_API_KEY')
    
#     if not groq_api_key:
#         print("❌ Error: Groq API key not found!")
#         print("Please provide your Groq API key using one of these methods:")
#         print("1. Create a .env file with: GROQ_API_KEY=your_api_key_here")
#         print("2. Set environment variable: export GROQ_API_KEY=your_api_key_here")
#         print("3. Use command line: --groq-key your_api_key_here")
#         print("\nGet your API key from: https://console.groq.com/")
#         sys.exit(1)
    
#     try:
#         # Initialize RAG system
#         print("🚀 Initializing Multilingual Urdu RAG System...")
#         rag_system = MultilingualRAG(groq_api_key, args.data)
        
#         # Load or create vector store
#         if args.rebuild_index or not os.path.exists(rag_system.faiss_index_file):
#             rag_system.create_vector_store()
#         else:
#             rag_system.load_vector_store()
        
#         # Start interactive chat
#         rag_system.interactive_chat()
        
#     except FileNotFoundError as e:
#         print(f"❌ Error: {e}")
#         print("Please make sure the OCR output file exists at the specified path.")
#     except Exception as e:
#         print(f"❌ Unexpected error: {e}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()





# import os
# import sys
# from pathlib import Path
# from typing import List, Optional
# import faiss
# import numpy as np
# import pickle
# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate
# import re
# import argparse
# from dotenv import load_dotenv
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class MultilingualRAG:
#     def __init__(self, groq_api_key: str, data_file: str = "output/output.txt", vector_store_path: str = "vector_store"):
#         """Initialize the multilingual RAG system"""
#         self.groq_api_key = groq_api_key
#         self.data_file = data_file
#         self.vector_store_path = vector_store_path
        
#         logger.info(f"Initializing RAG system with data file: {data_file}")
#         logger.info(f"Vector store path: {vector_store_path}")
        
#         # Initialize multilingual embedding model
#         logger.info("Loading multilingual embedding model...")
#         self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        
#         # Initialize Groq LLM with current production model
#         logger.info("Initializing Groq LLM...")
#         self.llm = ChatGroq(
#             temperature=0.7,
#             groq_api_key=groq_api_key,
#             model_name="llama-3.3-70b-versatile"  # Current production model
#         )
        
#         # Initialize components
#         self.vector_store = None
#         self.documents = []
#         self.chunks = []
        
#         # Vector store files
#         self.faiss_index_file = os.path.join(self.vector_store_path, "faiss_index.bin")
#         self.documents_file = os.path.join(self.vector_store_path, "documents.pkl")
#         self.chunks_file = os.path.join(self.vector_store_path, "chunks.pkl")
        
#         logger.info(f"FAISS index file: {self.faiss_index_file}")
#         logger.info(f"Documents file: {self.documents_file}")
#         logger.info(f"Chunks file: {self.chunks_file}")
        
#     def detect_language(self, text: str) -> str:
#         """Detect the language of input text"""
#         # Simple language detection based on script and patterns
        
#         # Check for Urdu script (Arabic/Persian characters)
#         urdu_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        
#         # Check for Roman Urdu patterns (common transliteration words)
#         roman_urdu_patterns = [
#             r'\b(kya|hai|hain|ka|ke|ki|ko|se|me|per|aur|ya|nahi|han|hum|tum|ap|woh|yeh|kaise|kahan|kab|kyun)\b',
#             r'\b(pahar|pani|ghar|sara|sare|wala|wali|wale|ata|ate|ati|karta|karte|karti)\b'
#         ]
        
#         if urdu_pattern.search(text):
#             return "urdu"
#         elif any(re.search(pattern, text.lower()) for pattern in roman_urdu_patterns):
#             return "roman_urdu"
#         else:
#             return "english"
    
#     def load_and_chunk_documents(self) -> List[Document]:
#         """Load and chunk the Urdu text documents"""
#         logger.info(f"Loading documents from {self.data_file}")
        
#         if not os.path.exists(self.data_file):
#             error_msg = f"Data file {self.data_file} not found!"
#             logger.error(error_msg)
#             raise FileNotFoundError(error_msg)
        
#         with open(self.data_file, 'r', encoding='utf-8') as f:
#             content = f.read()
        
#         # Split by page separators
#         pages = content.split('=' * 50)
#         documents = []
        
#         for i, page_content in enumerate(pages):
#             if page_content.strip():
#                 # Clean up the content
#                 cleaned_content = re.sub(r'=== Page \d+ ===', '', page_content).strip()
#                 if cleaned_content:
#                     documents.append(Document(
#                         page_content=cleaned_content,
#                         metadata={"page": i + 1, "source": self.data_file}
#                     ))
        
#         logger.info(f"Loaded {len(documents)} pages")
        
#         # Chunk the documents
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=500,
#             chunk_overlap=100,
#             separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
#         )
        
#         chunks = text_splitter.split_documents(documents)
#         logger.info(f"Created {len(chunks)} chunks")
        
#         self.documents = documents
#         self.chunks = chunks
#         return chunks
    
#     def create_vector_store(self):
#         """Create FAISS vector store from documents"""
#         logger.info("Creating vector store...")
        
#         if not self.chunks:
#             self.load_and_chunk_documents()
        
#         # Extract text from chunks
#         texts = [chunk.page_content for chunk in self.chunks]
        
#         # Create embeddings
#         logger.info("Creating embeddings...")
#         embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
#         # Create FAISS index
#         dimension = embeddings.shape[1]
#         index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
#         # Normalize embeddings for cosine similarity
#         faiss.normalize_L2(embeddings)
#         index.add(embeddings.astype('float32'))
        
#         self.vector_store = index
        
#         # Save vector store and metadata
#         self.save_vector_store()
        
#         logger.info(f"Vector store created with {index.ntotal} vectors")
    
#     def save_vector_store(self):
#         """Save vector store and associated data"""
#         logger.info(f"Saving vector store to {self.vector_store_path}")
        
#         # Ensure directory exists
#         os.makedirs(self.vector_store_path, exist_ok=True)
        
#         # Save FAISS index
#         try:
#             faiss.write_index(self.vector_store, self.faiss_index_file)
#             logger.info(f"FAISS index saved to {self.faiss_index_file}")
#         except Exception as e:
#             logger.error(f"Error saving FAISS index: {e}")
#             raise Exception(f"Error saving FAISS index: {e}")
        
#         # Save documents and chunks
#         try:
#             with open(self.documents_file, 'wb') as f:
#                 pickle.dump(self.documents, f)
#             logger.info(f"Documents saved to {self.documents_file}")
#         except Exception as e:
#             logger.error(f"Error saving documents: {e}")
#             raise Exception(f"Error saving documents: {e}")
        
#         try:
#             with open(self.chunks_file, 'wb') as f:
#                 pickle.dump(self.chunks, f)
#             logger.info(f"Chunks saved to {self.chunks_file}")
#         except Exception as e:
#             logger.error(f"Error saving chunks: {e}")
#             raise Exception(f"Error saving chunks: {e}")
        
#         logger.info("Vector store saved successfully!")
    
#     def load_vector_store(self):
#         """Load existing vector store"""
#         if not os.path.exists(self.faiss_index_file):
#             logger.info("No existing vector store found. Creating new one...")
#             self.create_vector_store()
#             return
        
#         logger.info("Loading existing vector store...")
        
#         # Load FAISS index
#         try:
#             self.vector_store = faiss.read_index(self.faiss_index_file)
#         except Exception as e:
#             logger.error(f"Error loading FAISS index: {e}")
#             raise Exception(f"Error loading FAISS index: {e}")
        
#         # Load documents and chunks
#         try:
#             with open(self.documents_file, 'rb') as f:
#                 self.documents = pickle.load(f)
#         except Exception as e:
#             logger.error(f"Error loading documents: {e}")
#             raise Exception(f"Error loading documents: {e}")
        
#         try:
#             with open(self.chunks_file, 'rb') as f:
#                 self.chunks = pickle.load(f)
#         except Exception as e:
#             logger.error(f"Error loading chunks: {e}")
#             raise Exception(f"Error loading chunks: {e}")
        
#         logger.info(f"Vector store loaded with {self.vector_store.ntotal} vectors")
    
#     def search_similar_chunks(self, query: str, k: int = 5) -> List[str]:
#         """Search for similar chunks using the query"""
#         # Create query embedding
#         query_embedding = self.embedding_model.encode([query])
#         faiss.normalize_L2(query_embedding)
        
#         # Search in vector store
#         scores, indices = self.vector_store.search(query_embedding.astype('float32'), k)
        
#         # Get relevant chunks
#         relevant_chunks = []
#         for idx in indices[0]:
#             if idx < len(self.chunks):
#                 relevant_chunks.append(self.chunks[idx].page_content)
        
#         return relevant_chunks
    
#     def preprocess_context_for_language(self, context: str, target_language: str) -> str:
#         """Preprocess context based on target language to maintain consistency"""
#         if target_language == "roman_urdu":
#             # For Roman Urdu responses, add instruction to convert any Urdu script
#             context_instruction = (
#                 "Note: The following context contains Urdu text. "
#                 "Please convert any Urdu script words to Roman Urdu in your response.\n\n"
#                 f"{context}"
#             )
#             return context_instruction
#         elif target_language == "english":
#             # For English responses, add instruction to translate Urdu words
#             context_instruction = (
#                 "Note: The following context contains Urdu text. "
#                 "Please translate any Urdu words to English in your response and provide English names/terms.\n\n"
#                 f"{context}"
#             )
#             return context_instruction
#         return context
    
#     def post_process_response(self, response: str, language: str) -> str:
#         """Post-process response to ensure language consistency"""
#         if language == "roman_urdu":
#             # Remove any remaining Urdu script characters if they slip through
#             urdu_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
#             if urdu_pattern.search(response):
#                 # If Urdu script found, add a note
#                 response += "\n\n(Note: Some original text was in Urdu script and may need manual conversion to Roman Urdu)"
#         elif language == "english":
#             # For English responses, check if Urdu script still exists
#             urdu_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
#             if urdu_pattern.search(response):
#                 response += "\n\n(Note: Some original names were in Urdu script - please refer to the English transliteration above)"
        
#         return response
    
#     def create_multilingual_prompt(self, language: str) -> ChatPromptTemplate:
#         """Create language-specific prompt templates"""
        
#         if language == "urdu":
#             template = """آپ ایک مددگار اردو اسسٹنٹ ہیں۔ دیے گئے سیاق و سباق کی بنیاد پر سوال کا جواب دیں۔
# جواب اردو میں دیں اور مکمل اور درست ہو۔

# سیاق و سباق:
# {context}

# سوال: {question}

# جواب:"""
        
#         elif language == "roman_urdu":
#             template = """Aap ek helpful Roman Urdu assistant hain. Diye gaye context ki base par sawal ka jawab dein.
# Jawab SIRF Roman Urdu mein dein. Urdu script (Arabic letters) ka istemaal BILKUL nahi karein.
# Sab kuch English/Roman letters mein likhein. Complete aur sahi jawab dein.

# IMPORTANT: Koi bhi Urdu script words ko Roman Urdu mein convert kar dein.

# Context:
# {context}

# Sawal: {question}

# Roman Urdu mein jawab:"""
        
#         else:  # English
#             template = """You are a helpful assistant. Answer the question based on the given context.
# Provide a complete and accurate answer in English. If the context contains Urdu names or words, 
# please provide their English transliteration (Roman script) or translation.

# Context:
# {context}

# Question: {question}

# Answer (in English only):"""
        
#         return ChatPromptTemplate.from_template(template)
    
#     def answer_query(self, query: str) -> str:
#         """Answer a multilingual query"""
#         # Detect language
#         language = self.detect_language(query)
#         logger.info(f"Detected language: {language}")
        
#         # Search for relevant chunks
#         relevant_chunks = self.search_similar_chunks(query, k=5)
#         context = "\n\n".join(relevant_chunks)
        
#         # Preprocess context for target language
#         context = self.preprocess_context_for_language(context, language)
        
#         # Create appropriate prompt
#         prompt_template = self.create_multilingual_prompt(language)
        
#         # Create and run chain using new LangChain syntax
#         try:
#             # Use the new RunnableSequence approach
#             chain = prompt_template | self.llm
#             response = chain.invoke({"context": context, "question": query})
            
#             # Extract content from the response
#             if hasattr(response, 'content'):
#                 final_response = response.content.strip()
#             else:
#                 final_response = str(response).strip()
            
#             # Post-process for language consistency
#             final_response = self.post_process_response(final_response, language)
#             return final_response
                
#         except Exception as e:
#             logger.error(f"Error generating response: {e}")
#             return f"Error generating response: {str(e)}"

# def main():
#     """Main function"""
#     # Load environment variables from .env file
#     load_dotenv()
    
#     parser = argparse.ArgumentParser(description='Multilingual Urdu RAG Application')
#     parser.add_argument('--data', default='output/output.txt', help='Path to the OCR text file')
#     parser.add_argument('--groq-key', help='Groq API key (overrides .env file and environment variable)')
#     parser.add_argument('--rebuild-index', action='store_true', help='Force rebuild of vector index')
    
#     args = parser.parse_args()
    
#     # Get Groq API key with priority: command line > .env file > environment variable
#     groq_api_key = args.groq_key or os.getenv('GROQ_API_KEY')
    
#     if not groq_api_key:
#         print("❌ Error: Groq API key not found!")
#         print("Please provide your Groq API key using one of these methods:")
#         print("1. Create a .env file with: GROQ_API_KEY=your_api_key_here")
#         print("2. Set environment variable: export GROQ_API_KEY=your_api_key_here")
#         print("3. Use command line: --groq-key your_api_key_here")
#         print("\nGet your API key from: https://console.groq.com/")
#         sys.exit(1)
    
#     try:
#         # Initialize RAG system
#         print("🚀 Initializing Multilingual Urdu RAG System...")
#         rag_system = MultilingualRAG(groq_api_key, args.data)
        
#         # Load or create vector store
#         if args.rebuild_index or not os.path.exists(rag_system.faiss_index_file):
#             rag_system.create_vector_store()
#         else:
#             rag_system.load_vector_store()
        
#         # Start interactive chat
#         rag_system.interactive_chat()
        
#     except FileNotFoundError as e:
#         print(f"❌ Error: {e}")
#         print("Please make sure the OCR output file exists at the specified path.")
#     except Exception as e:
#         print(f"❌ Unexpected error: {e}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()




import os
import sys
from pathlib import Path
from typing import List, Optional
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import re
import argparse
from dotenv import load_dotenv
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

class MultilingualRAG:
    def __init__(self, groq_api_key: str, data_file: str = "output/output.txt", vector_store_path: str = "vector_store"):
        """Initialize the multilingual RAG system"""
        self.groq_api_key = groq_api_key
        self.data_file = data_file
        self.vector_store_path = vector_store_path
        
        logger.info(f"Initializing RAG system with data file: {data_file}")
        logger.info(f"Vector store path: {vector_store_path}")
        
        # Initialize multilingual embedding model with device handling
        logger.info("Loading multilingual embedding model...")
        try:
            # Try to use the device we determined
            self.embedding_model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                device=device.type
            )
        except Exception as e:
            logger.warning(f"Failed to use device {device}: {e}")
            # Fall back to CPU
            try:
                self.embedding_model = SentenceTransformer(
                    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                    device='cpu'
                )
                logger.info("Falling back to CPU for embedding model")
            except Exception as e2:
                logger.error(f"Failed to load embedding model even on CPU: {e2}")
                raise Exception(f"Failed to load embedding model: {e2}")
        
        # Initialize Groq LLM with current production model
        logger.info("Initializing Groq LLM...")
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
        self.faiss_index_file = os.path.join(self.vector_store_path, "faiss_index.bin")
        self.documents_file = os.path.join(self.vector_store_path, "documents.pkl")
        self.chunks_file = os.path.join(self.vector_store_path, "chunks.pkl")
        
        logger.info(f"FAISS index file: {self.faiss_index_file}")
        logger.info(f"Documents file: {self.documents_file}")
        logger.info(f"Chunks file: {self.chunks_file}")
        
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
        logger.info(f"Loading documents from {self.data_file}")
        
        if not os.path.exists(self.data_file):
            error_msg = f"Data file {self.data_file} not found!"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
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
        
        logger.info(f"Loaded {len(documents)} pages")
        
        # Chunk the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        self.documents = documents
        self.chunks = chunks
        return chunks
    
    def create_vector_store(self):
        """Create FAISS vector store from documents"""
        logger.info("Creating vector store...")
        
        if not self.chunks:
            self.load_and_chunk_documents()
        
        # Extract text from chunks
        texts = [chunk.page_content for chunk in self.chunks]
        
        # Create embeddings with error handling
        logger.info("Creating embeddings...")
        try:
            # Use a more compatible approach for creating embeddings
            embeddings = []
            batch_size = 8  # Process in batches to avoid memory issues
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                try:
                    batch_embeddings = self.embedding_model.encode(batch_texts, convert_to_numpy=True)
                    embeddings.append(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error encoding batch {i//batch_size}: {e}")
                    # Create dummy embeddings for this batch
                    batch_embeddings = np.zeros((len(batch_texts), 768))  # Default embedding size
                    embeddings.append(batch_embeddings)
            
            # Combine all batch embeddings
            embeddings = np.vstack(embeddings)
            logger.info(f"Created embeddings with shape: {embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise Exception(f"Error creating embeddings: {e}")
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        self.vector_store = index
        
        # Save vector store and metadata
        self.save_vector_store()
        
        logger.info(f"Vector store created with {index.ntotal} vectors")
    
    def save_vector_store(self):
        """Save vector store and associated data"""
        logger.info(f"Saving vector store to {self.vector_store_path}")
        
        # Ensure directory exists
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Save FAISS index
        try:
            faiss.write_index(self.vector_store, self.faiss_index_file)
            logger.info(f"FAISS index saved to {self.faiss_index_file}")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise Exception(f"Error saving FAISS index: {e}")
        
        # Save documents and chunks
        try:
            with open(self.documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
            logger.info(f"Documents saved to {self.documents_file}")
        except Exception as e:
            logger.error(f"Error saving documents: {e}")
            raise Exception(f"Error saving documents: {e}")
        
        try:
            with open(self.chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            logger.info(f"Chunks saved to {self.chunks_file}")
        except Exception as e:
            logger.error(f"Error saving chunks: {e}")
            raise Exception(f"Error saving chunks: {e}")
        
        logger.info("Vector store saved successfully!")
    
    def load_vector_store(self):
        """Load existing vector store"""
        if not os.path.exists(self.faiss_index_file):
            logger.info("No existing vector store found. Creating new one...")
            self.create_vector_store()
            return
        
        logger.info("Loading existing vector store...")
        
        # Load FAISS index
        try:
            self.vector_store = faiss.read_index(self.faiss_index_file)
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise Exception(f"Error loading FAISS index: {e}")
        
        # Load documents and chunks
        try:
            with open(self.documents_file, 'rb') as f:
                self.documents = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise Exception(f"Error loading documents: {e}")
        
        try:
            with open(self.chunks_file, 'rb') as f:
                self.chunks = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
            raise Exception(f"Error loading chunks: {e}")
        
        logger.info(f"Vector store loaded with {self.vector_store.ntotal} vectors")
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[str]:
        """Search for similar chunks using the query"""
        # Create query embedding with error handling
        try:
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
        except Exception as e:
            logger.error(f"Error creating query embedding: {e}")
            raise Exception(f"Error creating query embedding: {e}")
        
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
        logger.info(f"Detected language: {language}")
        
        # Search for relevant chunks
        try:
            relevant_chunks = self.search_similar_chunks(query, k=5)
            context = "\n\n".join(relevant_chunks)
        except Exception as e:
            logger.error(f"Error searching for similar chunks: {e}")
            return f"Error searching for relevant information: {str(e)}"
        
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
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

def main():
    """Main function"""
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Multilingual Urdu RAG Application')
    parser.add_argument('--data', default='output/output.txt', help='Path to the OCR text file')
    parser.add_argument('--groq-key', help='Groq API key (overrides .env file and environment variable')
    parser.add_argument('--rebuild-index', action='store_true', help='Force rebuild of vector index')
    
    args = parser.parse_args()
    
    # Get Groq API key with priority: command line > .env file > environment variable
    groq_api_key = args.groq_key or os.getenv('GROQ_API_KEY')
    
    if not groq_api_key:
        print("❌ Error: Groq API key not found!")
        print("Please provide your Groq API key using one of these methods:")
        print("1. Create a .env file with: GROQ_API_KEY=your_api_key_here")
        print("2. Set environment variable: export GROQ_API_KEY=your_api_key_here")
        print("3. Use command line: --groq-key your_api_key_here")
        print("\nGet your API key from: https://console.groq.com/")
        sys.exit(1)
    
    try:
        # Initialize RAG system
        print("🚀 Initializing Multilingual Urdu RAG System...")
        rag_system = MultilingualRAG(groq_api_key, args.data)
        
        # Load or create vector store
        if args.rebuild_index or not os.path.exists(rag_system.faiss_index_file):
            rag_system.create_vector_store()
        else:
            rag_system.load_vector_store()
        
        # Start interactive chat
        rag_system.interactive_chat()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please make sure the OCR output file exists at the specified path.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()