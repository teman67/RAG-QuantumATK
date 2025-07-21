from flask import Flask, request, jsonify, render_template_string
import json
import os
from datetime import datetime
import logging
from quantumatk_rag import QuantumATKRAG  # Import the main RAG class

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG instance
rag_system = None

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuantumATK RAG Assistant</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .search-container {
            margin-bottom: 30px;
        }
        .search-box {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        .search-box:focus {
            outline: none;
            border-color: #667eea;
        }
        .search-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            cursor: pointer;
            margin-top: 10px;
            transition: transform 0.2s;
        }
        .search-btn:hover {
            transform: translateY(-2px);
        }
        .search-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .response-container {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        .response-text {
            line-height: 1.6;
            color: #333;
        }
        .sources {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
        }
        .source-item {
            margin-bottom: 10px;
            padding: 10px;
            background: white;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
        }
        .source-url {
            color: #667eea;
            text-decoration: none;
            font-weight: bold;
        }
        .source-url:hover {
            text-decoration: underline;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .loading::after {
            content: '';
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid #667eea;
            border-radius: 50%;
            border-top-color: transparent;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .examples {
            margin-top: 30px;
            padding: 20px;
            background: #e8f4f8;
            border-radius: 10px;
        }
        .example-question {
            background: white;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            cursor: pointer;
            border: 1px solid #d0d0d0;
            transition: background-color 0.2s;
        }
        .example-question:hover {
            background: #f0f0f0;
        }
        .stats {
            display: flex;
            justify-content: space-around;
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .stat-item {
            text-align: center;
        }
        .stat-number {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 QuantumATK RAG Assistant</h1>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-number" id="doc-count">-</div>
                <div class="stat-label">Documents</div>
            </div>
            <div class="stat-item">
                <div class="stat-number" id="kb-status">Loading...</div>
                <div class="stat-label">Knowledge Base</div>
            </div>
            <div class="stat-item">
                <div class="stat-number" id="query-count">0</div>
                <div class="stat-label">Queries Processed</div>
            </div>
        </div>
        
        <div class="search-container">
            <input type="text" class="search-box" id="questionInput" 
                   placeholder="Ask me anything about QuantumATK..." 
                   onkeypress="handleEnter(event)">
            <button class="search-btn" id="searchBtn" onclick="searchQuestion()">
                Search Knowledge Base
            </button>
        </div>
        
        <div id="responseContainer" style="display: none;"></div>
        
        <div class="examples">
            <h3>💡 Example Questions</h3>
            <div class="example-question" onclick="setQuestion('How do I perform a DFT calculation in QuantumATK?')">
                How do I perform a DFT calculation in QuantumATK?
            </div>
            <div class="example-question" onclick="setQuestion('What are the supported file formats for importing structures?')">
                What are the supported file formats for importing structures?
            </div>
            <div class="example-question" onclick="setQuestion('How to calculate band structure using QuantumATK?')">
                How to calculate band structure using QuantumATK?
            </div>
            <div class="example-question" onclick="setQuestion('What is the difference between ATK-DFT and ATK-SE?')">
                What is the difference between ATK-DFT and ATK-SE?
            </div>
            <div class="example-question" onclick="setQuestion('How to optimize a crystal structure?')">
                How to optimize a crystal structure?
            </div>
            <div class="example-question" onclick="setQuestion('What are the system requirements for QuantumATK?')">
                What are the system requirements for QuantumATK?
            </div>
        </div>
    </div>

    <script>
        let queryCount = 0;
        
        // Check system status on load
        window.onload = function() {
            checkSystemStatus();
        };
        
        function checkSystemStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('doc-count').textContent = data.document_count || 0;
                    document.getElementById('kb-status').textContent = data.status || 'Unknown';
                    if (data.status === 'Ready') {
                        document.getElementById('kb-status').style.color = '#28a745';
                    } else {
                        document.getElementById('kb-status').style.color = '#dc3545';
                    }
                })
                .catch(error => {
                    console.error('Error checking status:', error);
                    document.getElementById('kb-status').textContent = 'Error';
                    document.getElementById('kb-status').style.color = '#dc3545';
                });
        }
        
        function handleEnter(event) {
            if (event.key === 'Enter') {
                searchQuestion();
            }
        }
        
        function setQuestion(question) {
            document.getElementById('questionInput').value = question;
            searchQuestion();
        }
        
        function searchQuestion() {
            const question = document.getElementById('questionInput').value.trim();
            
            if (!question) {
                alert('Please enter a question!');
                return;
            }
            
            // Show loading
            const responseContainer = document.getElementById('responseContainer');
            responseContainer.style.display = 'block';
            responseContainer.innerHTML = '<div class="loading">Searching knowledge base...</div>';
            
            // Disable search button
            const searchBtn = document.getElementById('searchBtn');
            searchBtn.disabled = true;
            searchBtn.textContent = 'Searching...';
            
            // Make API call
            fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question })
            })
            .then(response => response.json())
            .then(data => {
                displayResponse(data);
                queryCount++;
                document.getElementById('query-count').textContent = queryCount;
            })
            .catch(error => {
                console.error('Error:', error);
                displayError('An error occurred while processing your question.');
            })
            .finally(() => {
                // Re-enable search button
                searchBtn.disabled = false;
                searchBtn.textContent = 'Search Knowledge Base';
            });
        }
        
        function displayResponse(data) {
            const responseContainer = document.getElementById('responseContainer');
            
            if (data.error) {
                displayError(data.error);
                return;
            }
            
            let html = `
                <div class="response-container">
                    <div class="response-text">
                        ${data.answer.replace(/\n/g, '<br>')}
                    </div>
            `;
            
            if (data.sources && data.sources.length > 0) {
                html += '<div class="sources"><h4>📚 Sources:</h4>';
                data.sources.forEach(source => {
                    html += `
                        <div class="source-item">
                            <a href="${source.url}" class="source-url" target="_blank">
                                ${source.title}
                            </a>
                            <div style="font-size: 12px; color: #666; margin-top: 5px;">
                                Section: ${source.section} | Relevance: ${(source.score * 100).toFixed(1)}%
                            </div>
                        </div>
                    `;
                });
                html += '</div>';
            }
            
            html += '</div>';
            responseContainer.innerHTML = html;
        }
        
        function displayError(message) {
            const responseContainer = document.getElementById('responseContainer');
            responseContainer.innerHTML = `
                <div class="response-container" style="border-left-color: #dc3545;">
                    <div class="response-text" style="color: #dc3545;">
                        ❌ ${message}
                    </div>
                </div>
            `;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    """Get system status"""
    global rag_system
    
    if rag_system is None:
        return jsonify({
            'status': 'Not Initialized',
            'document_count': 0,
            'message': 'RAG system not initialized. Please run setup first.'
        })
    
    try:
        doc_count = len(rag_system.vector_store.documents)
        return jsonify({
            'status': 'Ready',
            'document_count': doc_count,
            'message': 'System ready for queries'
        })
    except Exception as e:
        return jsonify({
            'status': 'Error',
            'document_count': 0,
            'message': str(e)
        })

@app.route('/query', methods=['POST'])
def query():
    """Handle search queries"""
    global rag_system
    
    if rag_system is None:
        return jsonify({'error': 'RAG system not initialized'})
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'No question provided'})
        
        # Get answer from RAG system
        answer = rag_system.query(question)
        
        # Get source documents for transparency
        results = rag_system.vector_store.search(question, k=3)
        sources = []
        
        for doc, score in results:
            sources.append({
                'title': doc.title,
                'url': doc.url,
                'section': doc.section,
                'score': score
            })
        
        return jsonify({
            'answer': answer,
            'sources': sources,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': f'Error processing query: {str(e)}'})

@app.route('/setup', methods=['POST'])
def setup():
    """Setup the RAG system"""
    global rag_system
    
    try:
        data = request.get_json()
        openai_api_key = data.get('openai_api_key')
        
        if not openai_api_key:
            return jsonify({'error': 'OpenAI API key required'})
        
        # Initialize RAG system
        rag_system = QuantumATKRAG(openai_api_key=openai_api_key)
        
        # Check if knowledge base exists
        kb_path = "quantumatk_kb/vectorstore"
        if os.path.exists(f"{kb_path}.index") and os.path.exists(f"{kb_path}.docs"):
            # Load existing knowledge base
            rag_system.load_knowledge_base(kb_path)
            doc_count = len(rag_system.vector_store.documents)
            
            return jsonify({
                'status': 'success',
                'message': f'Loaded existing knowledge base with {doc_count} documents',
                'document_count': doc_count
            })
        else:
            return jsonify({
                'status': 'no_kb',
                'message': 'No existing knowledge base found. Please run build_kb endpoint first.'
            })
            
    except Exception as e:
        logger.error(f"Error setting up RAG system: {e}")
        return jsonify({'error': f'Setup error: {str(e)}'})

@app.route('/build_kb', methods=['POST'])
def build_knowledge_base():
    """Build knowledge base from QuantumATK sources"""
    global rag_system
    
    if rag_system is None:
        return jsonify({'error': 'RAG system not initialized. Run setup first.'})
    
    try:
        data = request.get_json()
        max_pages = data.get('max_pages', 100)
        
        base_urls = [
            "https://docs.quantumatk.com/",
            
        ]
        
        # Build knowledge base
        logger.info("Starting knowledge base construction...")
        doc_count = rag_system.build_knowledge_base(base_urls, max_pages)
        
        # Save knowledge base
        kb_path = "quantumatk_kb/vectorstore"
        rag_system.save_knowledge_base(kb_path)
        
        return jsonify({
            'status': 'success',
            'message': f'Knowledge base built with {doc_count} documents',
            'document_count': doc_count
        })
        
    except Exception as e:
        logger.error(f"Error building knowledge base: {e}")
        return jsonify({'error': f'Build error: {str(e)}'})

@app.route('/rebuild_kb', methods=['POST'])
def rebuild_knowledge_base():
    """Rebuild knowledge base from scratch"""
    global rag_system
    
    if rag_system is None:
        return jsonify({'error': 'RAG system not initialized. Run setup first.'})
    
    try:
        # Clear existing knowledge base
        rag_system.vector_store = VectorStore()
        
        data = request.get_json()
        max_pages = data.get('max_pages', 100)
        
        base_urls = [
            "https://docs.quantumatk.com/",
            
        ]
        
        # Rebuild knowledge base
        logger.info("Rebuilding knowledge base from scratch...")
        doc_count = rag_system.build_knowledge_base(base_urls, max_pages)
        
        # Save knowledge base
        kb_path = "quantumatk_kb/vectorstore"
        rag_system.save_knowledge_base(kb_path)
        
        return jsonify({
            'status': 'success',
            'message': f'Knowledge base rebuilt with {doc_count} documents',
            'document_count': doc_count
        })
        
    except Exception as e:
        logger.error(f"Error rebuilding knowledge base: {e}")
        return jsonify({'error': f'Rebuild error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)