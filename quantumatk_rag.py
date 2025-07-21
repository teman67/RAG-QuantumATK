import os
import requests
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup
import pickle
import json
from urllib.parse import urljoin, urlparse
import re
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# For embeddings and vector search
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import tiktoken

# For LLM integration
from openai import OpenAI
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Represents a document chunk with metadata"""
    content: str
    url: str
    title: str
    doc_type: str  # 'docs' or 'forum'
    section: str
    chunk_id: str
    metadata: Dict[str, Any]

class QuantumATKScraper:
    """Scrapes QuantumATK documentation and forum content"""
    
    def __init__(self, base_urls: List[str], max_pages: int = 1000):
        self.base_urls = base_urls
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.visited_urls = set()
        self.documents = []
        
    def is_valid_url(self, url: str, base_url: str) -> bool:
        """Check if URL is valid for scraping"""
        if not url or url in self.visited_urls:
            return False
        
        parsed = urlparse(url)
        base_parsed = urlparse(base_url)
        
        # Same domain check
        if parsed.netloc != base_parsed.netloc:
            return False
        
        # Skip certain file types and fragments
        skip_extensions = ['.pdf', '.zip', '.tar', '.gz', '.jpg', '.png', '.gif']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        # Skip fragments and queries for forum
        if 'forum' in base_url and ('#' in url or '?' in url):
            return False
            
        return True
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract relevant content from HTML"""
        content_data = {
            'title': '',
            'content': '',
            'section': '',
            'doc_type': 'docs' if 'docs' in url else 'forum'
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            content_data['title'] = title_tag.get_text().strip()
        
        # For documentation pages
        if 'docs' in url:
            # Remove navigation, headers, footers
            for tag in soup.find_all(['nav', 'header', 'footer', 'aside']):
                tag.decompose()
            
            # Extract main content
            main_content = soup.find('main') or soup.find('div', class_='content')
            if main_content:
                content_data['content'] = main_content.get_text(separator='\n').strip()
            else:
                content_data['content'] = soup.get_text(separator='\n').strip()
                
            # Extract section from URL or heading
            if '/manual/' in url:
                content_data['section'] = 'manual'
            elif '/tutorials/' in url:
                content_data['section'] = 'tutorials'
            elif '/guides/' in url:
                content_data['section'] = 'guides'
            else:
                content_data['section'] = 'general'
        
        # For forum pages
        else:
            # Extract post content
            posts = soup.find_all(['div', 'article'], class_=['post', 'message', 'topic'])
            post_texts = []
            
            for post in posts:
                # Remove quotes and signatures
                for quote in post.find_all(class_=['quote', 'signature']):
                    quote.decompose()
                
                text = post.get_text(separator='\n').strip()
                if text and len(text) > 50:  # Filter out very short posts
                    post_texts.append(text)
            
            content_data['content'] = '\n\n'.join(post_texts[:5])  # Limit to first 5 posts
            content_data['section'] = 'forum'
        
        return content_data
    
    def scrape_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape a single URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            content_data = self.extract_content(soup, url)
            
            if content_data['content']:
                content_data['url'] = url
                return content_data
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            
        return None
    
    def find_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Find all valid links on a page"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            if self.is_valid_url(full_url, base_url):
                links.append(full_url)
        
        return links
    
    def scrape_all(self) -> List[Dict[str, Any]]:
        """Scrape all pages from base URLs"""
        to_visit = list(self.base_urls)
        scraped_data = []
        
        while to_visit and len(scraped_data) < self.max_pages:
            current_batch = to_visit[:10]  # Process in batches
            to_visit = to_visit[10:]
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(self.scrape_url, url): url for url in current_batch}
                
                for future in as_completed(futures):
                    url = futures[future]
                    if url in self.visited_urls:
                        continue
                        
                    self.visited_urls.add(url)
                    
                    try:
                        data = future.result()
                        if data:
                            scraped_data.append(data)
                            
                            # Find more links if it's a docs page
                            if 'docs' in url:
                                response = self.session.get(url, timeout=10)
                                soup = BeautifulSoup(response.content, 'html.parser')
                                new_links = self.find_links(soup, url)
                                to_visit.extend(new_links)
                                
                    except Exception as e:
                        logger.error(f"Error processing {url}: {e}")
            
            logger.info(f"Scraped {len(scraped_data)} pages so far...")
            time.sleep(1)  # Be respectful
        
        return scraped_data

class TextChunker:
    """Chunks text into manageable pieces for embedding"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If paragraph is too long, split it further
            if self.count_tokens(paragraph) > self.chunk_size:
                # Split by sentences
                sentences = re.split(r'[.!?]+', paragraph)
                for sentence in sentences:
                    if self.count_tokens(current_chunk + sentence) > self.chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            chunks.append(sentence[:self.chunk_size])
                    else:
                        current_chunk += sentence + ". "
            else:
                if self.count_tokens(current_chunk + paragraph) > self.chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = paragraph
                    else:
                        chunks.append(paragraph)
                else:
                    current_chunk += paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def create_documents(self, scraped_data: List[Dict[str, Any]]) -> List[Document]:
        """Create Document objects from scraped data"""
        documents = []
        
        for data in scraped_data:
            chunks = self.chunk_text(data['content'])
            
            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{data['url']}-{i}".encode()).hexdigest()
                
                doc = Document(
                    content=chunk,
                    url=data['url'],
                    title=data['title'],
                    doc_type=data['doc_type'],
                    section=data['section'],
                    chunk_id=doc_id,
                    metadata={
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'content_length': len(chunk)
                    }
                )
                documents.append(doc)
        
        return documents

class VectorStore:
    """Vector store for document embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for similarity
        self.documents = []
    
    def add_documents(self, documents: List[Document]):
        """Add documents to vector store"""
        self.documents.extend(documents)
        
        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype(np.float32))
    
    def search(self, query: str, k: int = 5) -> List[tuple]:
        """Search for similar documents"""
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save(self, path: str):
        """Save vector store to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save documents
        with open(f"{path}.docs", 'wb') as f:
            pickle.dump(self.documents, f)
    
    def load(self, path: str):
        """Load vector store from disk"""
        self.index = faiss.read_index(f"{path}.index")
        
        with open(f"{path}.docs", 'rb') as f:
            self.documents = pickle.load(f)

class QuantumATKRAG:
    """Main RAG system for QuantumATK"""
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model_name = model_name
        self.vector_store = VectorStore()
        self.system_prompt = """You are a helpful assistant specializing in QuantumATK, a computational platform for atomic-scale calculations and materials science. 

You have access to relevant documentation and forum discussions. When answering questions:
1. Provide accurate, technical information based on the retrieved context
2. Include specific code examples when relevant
3. Reference the source documentation or forum posts when helpful
4. If the question is about a specific feature, explain both how to use it and its theoretical background
5. For troubleshooting questions, provide step-by-step solutions
6. Always maintain scientific accuracy and precision

If you cannot find relevant information in the provided context, clearly state this and suggest where the user might find more information."""
    
    def query(self, question: str, k: int = 5) -> str:
        """Answer a question using RAG"""
        # Retrieve relevant documents
        results = self.vector_store.search(question, k=k)
        
        if not results:
            return "I couldn't find relevant information in the QuantumATK documentation and forum. Please try rephrasing your question or check the official documentation at https://docs.quantumatk.com/"
        
        # Prepare context
        context_parts = []
        for doc, score in results:
            context_parts.append(f"Source: {doc.url} ({doc.section})\nTitle: {doc.title}\nContent: {doc.content}\nRelevance Score: {score:.3f}\n")
        
        context = "\n" + "="*50 + "\n".join(context_parts)
        
        # Generate response
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nRelevant Context:\n{context}"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def build_knowledge_base(self, base_urls: List[str], max_pages: int = 500):
        """Build the knowledge base from QuantumATK sources"""
        logger.info("Starting web scraping...")
        
        # Scrape content
        scraper = QuantumATKScraper(base_urls, max_pages)
        scraped_data = scraper.scrape_all()
        
        logger.info(f"Scraped {len(scraped_data)} pages")
        
        # Chunk documents
        chunker = TextChunker()
        documents = chunker.create_documents(scraped_data)
        
        logger.info(f"Created {len(documents)} document chunks")
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        
        logger.info("Knowledge base built successfully!")
        
        return len(documents)
    
    def save_knowledge_base(self, path: str):
        """Save the knowledge base"""
        self.vector_store.save(path)
    
    def load_knowledge_base(self, path: str):
        """Load the knowledge base"""
        self.vector_store.load(path)

def main():
    """Example usage"""
    # Initialize the RAG system
    rag = QuantumATKRAG(openai_api_key="your-openai-api-key")
    
    # Define URLs to scrape
    base_urls = [
        "https://docs.quantumatk.com/",
        
    ]
    
    # Build knowledge base (run once)
    print("Building knowledge base...")
    num_docs = rag.build_knowledge_base(base_urls, max_pages=200)
    print(f"Knowledge base built with {num_docs} documents")
    
    # Save knowledge base
    rag.save_knowledge_base("quantumatk_kb/vectorstore")
    
    # Example queries
    questions = [
        "How do I perform a DFT calculation in QuantumATK?",
        "What are the supported file formats for importing structures?",
        "How to calculate band structure using QuantumATK?",
        "What is the difference between ATK-DFT and ATK-SE?",
        "How to optimize a crystal structure?",
        "What are the system requirements for QuantumATK?"
    ]
    
    print("\nExample Q&A:")
    for question in questions:
        print(f"\nQ: {question}")
        answer = rag.query(question)
        print(f"A: {answer}")
        print("-" * 80)

if __name__ == "__main__":
    main()