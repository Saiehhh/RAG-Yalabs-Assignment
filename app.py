
from flask import Flask, request, jsonify, render_template_string
import pdfplumber
import re
import nltk
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from flask_cors import CORS
import logging
import threading
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data (run once)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    logger.warning(f"NLTK download failed: {e}, but continuing...")

app = Flask(__name__)
CORS(app)

# Global variables for models (loaded once)
sentence_model = None
chunks = None
index = None
system_ready = False
initialization_error = None

# Modern Chatbot HTML Template with proper Jinja2 escaping
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OASIS-E1 Chatbot</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <style>
    .chat-container {{ '{' }}
      height: 70vh;
      max-height: 600px;
    {{ '}' }}
    .message-animation {{ '{' }}
      animation: fadeInUp 0.3s ease-out;
    {{ '}' }}
    @keyframes fadeInUp {{ '{' }}
      from {{ '{' }}
        opacity: 0;
        transform: translateY(10px);
      {{ '}' }}
      to {{ '{' }}
        opacity: 1;
        transform: translateY(0);
      {{ '}' }}
    {{ '}' }}
    .typing-indicator {{ '{' }}
      animation: pulse 1.5s ease-in-out infinite;
    {{ '}' }}
  </style>
</head>
<body class="bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 min-h-screen">
  <div id="root"></div>
  <script type="text/babel">
    const {{ '{' }} useState, useRef, useEffect {{ '}' }} = React;
    function App() {{ '{' }}
      const [messages, setMessages] = useState([]);
      const [currentMessage, setCurrentMessage] = useState('');
      const [isLoading, setIsLoading] = useState(false);
      const [systemStatus, setSystemStatus] = useState('Initializing...');
      const [isReady, setIsReady] = useState(false);
      const messagesEndRef = useRef(null);
      const inputRef = useRef(null);
      
      const scrollToBottom = () => {{ '{' }}
        messagesEndRef.current?.scrollIntoView({{ '{' }} behavior: "smooth" {{ '}' }});
      {{ '}' }};
      
      useEffect(() => {{ '{' }}
        scrollToBottom();
      {{ '}' }}, [messages]);

      useEffect(() => {{ '{' }}
        // Check system status periodically
        const checkStatus = async () => {{ '{' }}
          try {{ '{' }}
            const response = await fetch('/status');
            const data = await response.json();
            setSystemStatus(data.status);
            
            if (data.status === 'Ready') {{ '{' }}
              setIsReady(true);
              if (messages.length === 0) {{ '{' }}
                setMessages([{{ '{' }}
                  id: 1,
                  text: "Hello! I'm your OASIS-E1 assistant. I can help you with questions about the OASIS-E1 manual. What would you like to know?",
                  isBot: true,
                  timestamp: new Date().toLocaleTimeString([], {{ '{' }}hour: '2-digit', minute:'2-digit'{{ '}' }})
                {{ '}' }}]);
              {{ '}' }}
            {{ '}' }}
          {{ '}' }} catch (error) {{ '{' }}
            console.error('Status check failed:', error);
            setSystemStatus('Error checking status');
          {{ '}' }}
        {{ '}' }};

        checkStatus();
        const interval = setInterval(checkStatus, 3000); // Check every 3 seconds
        return () => clearInterval(interval);
      {{ '}' }}, [messages.length]);

      const handleSendMessage = async () => {{ '{' }}
        if (!currentMessage.trim() || isLoading || !isReady) return;
        
        const userMessage = {{ '{' }}
          id: Date.now(),
          text: currentMessage,
          isBot: false,
          timestamp: new Date().toLocaleTimeString([], {{ '{' }}hour: '2-digit', minute:'2-digit'{{ '}' }})
        {{ '}' }};
        
        setMessages(prev => [...prev, userMessage]);
        setCurrentMessage('');
        setIsLoading(true);
        
        try {{ '{' }}
          const response = await fetch('/ask', {{ '{' }}
            method: 'POST',
            headers: {{ '{' }} 'Content-Type': 'application/json' {{ '}' }},
            body: JSON.stringify({{ '{' }} query: currentMessage {{ '}' }}),
          {{ '}' }});
          
          if (!response.ok) throw new Error('Server error');
          
          const data = await response.json();
          
          const botMessage = {{ '{' }}
            id: Date.now() + 1,
            text: data.answer || 'I apologize, but I could not find an answer to your question.',
            isBot: true,
            timestamp: new Date().toLocaleTimeString([], {{ '{' }}hour: '2-digit', minute:'2-digit'{{ '}' }})
          {{ '}' }};
          setMessages(prev => [...prev, botMessage]);
        {{ '}' }} catch (error) {{ '{' }}
          const errorMessage = {{ '{' }}
            id: Date.now() + 1,
            text: 'Sorry, I encountered an error. Please try again.',
            isBot: true,
            timestamp: new Date().toLocaleTimeString([], {{ '{' }}hour: '2-digit', minute:'2-digit'{{ '}' }})
          {{ '}' }};
          setMessages(prev => [...prev, errorMessage]);
        {{ '}' }}
        
        setIsLoading(false);
        inputRef.current?.focus();
      {{ '}' }};

      const handleKeyPress = (e) => {{ '{' }}
        if (e.key === 'Enter' && !e.shiftKey) {{ '{' }}
          e.preventDefault();
          handleSendMessage();
        {{ '}' }}
      {{ '}' }};

      const clearChat = () => {{ '{' }}
        if (isReady) {{ '{' }}
          setMessages([{{ '{' }}
            id: 1,
            text: "Hello! I'm your OASIS-E1 assistant. I can help you with questions about the OASIS-E1 manual. What would you like to know?",
            isBot: true,
            timestamp: new Date().toLocaleTimeString([], {{ '{' }}hour: '2-digit', minute:'2-digit'{{ '}' }})
          {{ '}' }}]);
        {{ '}' }}
      {{ '}' }};

      const quickQuestions = [
        "What is OASIS-E1?",
        "Tell me about discharge planning",
        "How do I assess functional status?",
        "Explain medication management",
        "What are the documentation requirements?"
      ];

      // Show initialization screen
      if (!isReady) {{ '{' }}
        return (
          <div className="flex flex-col h-screen max-w-4xl mx-auto bg-white shadow-2xl">
            <div className="bg-gradient-to-r from-blue-600 to-indigo-700 p-4 text-white">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-white bg-opacity-20 rounded-full flex items-center justify-center">
                  <svg className="w-6 h-6 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                </div>
                <div>
                  <h1 className="text-xl font-bold">OASIS-E1 Chatbot</h1>
                  <p className="text-blue-100 text-sm">Status: {{ '{' }}systemStatus{{ '}' }}</p>
                </div>
              </div>
            </div>
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                <h2 className="text-xl font-semibold text-gray-700 mb-2">Initializing System</h2>
                <p className="text-gray-500">{{ '{' }}systemStatus{{ '}' }}</p>
                <p className="text-sm text-gray-400 mt-2">This may take a few moments...</p>
              </div>
            </div>
          </div>
        );
      {{ '}' }}

      return (
        <div className="flex flex-col h-screen max-w-4xl mx-auto bg-white shadow-2xl">
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-600 to-indigo-700 p-4 text-white">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-white bg-opacity-20 rounded-full flex items-center justify-center">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <div>
                  <h1 className="text-xl font-bold">OASIS-E1 Chatbot</h1>
                  <p className="text-blue-100 text-sm">Status: {{ '{' }}systemStatus{{ '}' }}</p>
                </div>
              </div>
              <button 
                onClick={{ '{' }}clearChat{{ '}' }}
                className="px-4 py-2 bg-white bg-opacity-20 hover:bg-opacity-30 rounded-lg text-sm font-medium transition-all duration-200"
              >
                Clear Chat
              </button>
            </div>
          </div>
          
          {/* Messages Container */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 chat-container">
            {{ '{' }}messages.map((message) => (
              <div
                key={{ '{' }}message.id{{ '}' }}
                className={{ '{' }}`flex ${{ '{' }}message.isBot ? 'justify-start' : 'justify-end'{{ '}' }} message-animation`{{ '}' }}
              >
                <div className={{ '{' }}`flex max-w-xs lg:max-w-md xl:max-w-lg ${{ '{' }}message.isBot ? 'flex-row' : 'flex-row-reverse'{{ '}' }} items-end space-x-2`{{ '}' }}>
                  <div className={{ '{' }}`w-8 h-8 rounded-full flex items-center justify-center ${{ '{' }}message.isBot ? 'bg-blue-500 text-white' : 'bg-gray-500 text-white'{{ '}' }}`{{ '}' }}>
                    {{ '{' }}message.isBot ? (
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z" />
                      </svg>
                    ) : (
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
                      </svg>
                    ){{ '}' }}
                  </div>
                  <div className={{ '{' }}`px-4 py-3 rounded-2xl ${{ '{' }}message.isBot ? 'bg-gray-100 text-gray-800' : 'bg-blue-600 text-white'{{ '}' }} shadow-sm`{{ '}' }}>
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{{ '{' }}message.text{{ '}' }}</p>
                    <p className={{ '{' }}`text-xs mt-2 ${{ '{' }}message.isBot ? 'text-gray-500' : 'text-blue-100'{{ '}' }}`{{ '}' }}>
                      {{ '{' }}message.timestamp{{ '}' }}
                    </p>
                  </div>
                </div>
              </div>
            )){{ '}' }}
            
            {{ '{' }}isLoading && (
              <div className="flex justify-start message-animation">
                <div className="flex items-end space-x-2">
                  <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z" />
                    </svg>
                  </div>
                  <div className="bg-gray-100 px-4 py-3 rounded-2xl">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full typing-indicator"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full typing-indicator" style={{ '{' }}{{ '{' }}animationDelay: '0.2s'{{ '}' }}{{ '}' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full typing-indicator" style={{ '{' }}{{ '{' }}animationDelay: '0.4s'{{ '}' }}{{ '}' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            ){{ '}' }}
            <div ref={{ '{' }}messagesEndRef{{ '}' }} />
          </div>
          
          {/* Quick Questions */}
          {{ '{' }}messages.length === 1 && (
            <div className="px-4 py-2 bg-gray-50 border-t">
              <p className="text-sm text-gray-600 mb-2">Quick questions to get started:</p>
              <div className="flex flex-wrap gap-2">
                {{ '{' }}quickQuestions.map((question, index) => (
                  <button
                    key={{ '{' }}index{{ '}' }}
                    onClick={{ '{' }}() => setCurrentMessage(question){{ '}' }}
                    className="px-3 py-1 bg-blue-100 text-blue-700 text-xs rounded-full hover:bg-blue-200 transition-colors duration-200"
                  >
                    {{ '{' }}question{{ '}' }}
                  </button>
                )){{ '}' }}
              </div>
            </div>
          ){{ '}' }}
          
          {/* Input Area */}
          <div className="p-4 bg-white border-t border-gray-200">
            <div className="flex space-x-3">
              <textarea
                ref={{ '{' }}inputRef{{ '}' }}
                value={{ '{' }}currentMessage{{ '}' }}
                onChange={{ '{' }}(e) => setCurrentMessage(e.target.value){{ '}' }}
                onKeyPress={{ '{' }}handleKeyPress{{ '}' }}
                placeholder="Type your question about OASIS-E1..."
                className="flex-1 p-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                rows="1"
                style={{ '{' }}{{ '{' }}minHeight: '44px', maxHeight: '120px'{{ '}' }}{{ '}' }}
                disabled={{ '{' }}isLoading{{ '}' }}
              />
              <button
                onClick={{ '{' }}handleSendMessage{{ '}' }}
                disabled={{ '{' }}isLoading || !currentMessage.trim(){{ '}' }}
                className="px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors duration-200 flex items-center justify-center"
              >
                {{ '{' }}isLoading ? (
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                ) : (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                ){{ '}' }}
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-2 text-center">
              Press Enter to send • Shift + Enter for new line
            </p>
          </div>
        </div>
      );
    {{ '}' }}
    ReactDOM.render(<App />, document.getElementById('root'));
  </script>
</body>
</html>
'''

def get_pdf_path():
    """Find the PDF file in the current directory"""
    possible_names = [
        "draft-oasis-e1-manual-04-28-2024.pdf",
        "oasis-e1-manual.pdf", 
        "manual.pdf"
    ]
    
    for name in possible_names:
        if os.path.exists(name):
            logger.info(f"Found PDF: {name}")
            return name
    
    # List all files in current directory
    files = os.listdir(".")
    pdf_files = [f for f in files if f.endswith('.pdf')]
    
    if pdf_files:
        logger.info(f"Using first PDF found: {pdf_files[0]}")
        return pdf_files[0]
    
    logger.error("No PDF file found!")
    return None

def initialize_models():
    """Initialize all models and load/build index"""
    global sentence_model, chunks, index, system_ready, initialization_error
    
    try:
        logger.info("Starting model initialization...")
        
        # Initialize sentence transformer model
        logger.info("Loading sentence transformer...")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence transformer loaded successfully")
        
        # Load or build index
        if os.path.exists('faiss_index.bin') and os.path.exists('embeddings.pkl'):
            logger.info("Loading existing index...")
            load_index()
        else:
            logger.info("Building new index...")
            pdf_path = get_pdf_path()
            if pdf_path:
                build_new_index(pdf_path)
            else:
                raise Exception("No PDF file found to build index")
        
        system_ready = True
        logger.info("System initialization completed successfully!")
        
    except Exception as e:
        initialization_error = str(e)
        logger.error(f"Failed to initialize: {e}")
        system_ready = False

def load_index():
    """Load existing index and chunks"""
    global chunks, index
    
    try:
        # Load chunks
        with open('embeddings.pkl', 'rb') as f:
            data = pickle.load(f)
            chunks = data['chunks']
        
        # Load FAISS index
        index = faiss.read_index('faiss_index.bin')
        logger.info(f"Index loaded with {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        raise

def build_new_index(pdf_path):
    """Build new index from PDF"""
    global chunks, index
    
    try:
        logger.info("Parsing PDF...")
        raw_chunks = parse_pdf(pdf_path)
        
        logger.info("Creating refined chunks...")
        refined_chunks = chunk_with_overlap(raw_chunks)
        
        logger.info("Building FAISS index...")
        index, chunks = build_index(refined_chunks)
        
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        raise

def parse_pdf(pdf_path):
    """Parse PDF and extract chunks"""
    chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Processing {total_pages} pages...")
            
            for page_num, page in enumerate(pdf.pages, start=1):
                if page_num % 10 == 0:
                    logger.info(f"Processed {page_num}/{total_pages} pages")
                
                text = page.extract_text()
                if text:
                    # Clean text
                    text = re.sub(r'\nPage \d+\n', '', text)
                    text = re.sub(r'OASIS-E1\s+Effective 01/01/2025\s+Centers for Medicare & Medicaid Services\s+\w+', '', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    if len(text) > 100:  # Only keep substantial text
                        chunks.append({
                            "text": text, 
                            "metadata": {"page": page_num}
                        })
        
        logger.info(f"Extracted {len(chunks)} chunks from PDF")
        return chunks
        
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        raise

def chunk_with_overlap(chunks, max_length=1000, overlap=200):
    """Create overlapping chunks"""
    refined_chunks = []
    
    for i, chunk in enumerate(chunks):
        text = chunk['text']
        
        # If text is short enough, keep as is
        if len(text) <= max_length:
            refined_chunks.append(chunk)
            continue
            
        # Split long text into overlapping chunks
        start = 0
        while start < len(text):
            end = start + max_length
            chunk_text = text[start:end]
            
            refined_chunks.append({
                "text": chunk_text,
                "metadata": chunk['metadata']
            })
            
            start = end - overlap
            if start >= len(text):
                break
    
    logger.info(f"Created {len(refined_chunks)} refined chunks")
    return refined_chunks

def build_index(chunks, index_path='faiss_index.bin', embeddings_path='embeddings.pkl'):
    """Build FAISS index from chunks"""
    try:
        texts = [c['text'] for c in chunks]
        logger.info(f"Encoding {len(texts)} texts...")
        
        # Encode texts in batches
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = sentence_model.encode(batch_texts, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Encoded {i + len(batch_texts)}/{len(texts)} texts")
        
        embeddings = np.array(embeddings)
        
        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype('float32'))
        
        # Save index and chunks
        faiss.write_index(index, index_path)
        with open(embeddings_path, 'wb') as f:
            pickle.dump({'chunks': chunks}, f)
        
        logger.info(f"Index built and saved with {len(chunks)} chunks!")
        return index, chunks
        
    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise

def retrieve(query, top_k=5):
    """Retrieve relevant chunks for query"""
    try:
        if not system_ready or sentence_model is None or index is None or chunks is None:
            return [{"text": "System is still initializing. Please wait a moment and try again.", "metadata": {"page": 0}}]
        
        # Encode query
        query_emb = sentence_model.encode([query])
        
        # Search FAISS index  
        scores, indices = index.search(query_emb.astype('float32'), top_k)
        
        # Get top chunks
        top_chunks = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks) and scores[0][i] < 2.0:  # Distance threshold
                top_chunks.append(chunks[idx])
        
        if not top_chunks:
            return [{"text": "I couldn't find relevant information in the OASIS-E1 manual for your question.", "metadata": {"page": 0}}]
        
        return top_chunks
        
    except Exception as e:
        logger.error(f"Error in retrieve: {e}")
        return [{"text": "Error retrieving information. Please try again.", "metadata": {"page": 0}}]

def generate_answer(query, contexts):
    """Generate answer from retrieved contexts"""
    try:
        if not contexts:
            return "I couldn't find relevant information for your question."
            
        # Handle special messages
        first_text = contexts[0]['text']
        if "System is still initializing" in first_text or "Error" in first_text or "couldn't find relevant" in first_text:
            return first_text
        
        # Create answer from contexts
        answer_parts = []
        seen_content = set()
        
        for context in contexts[:3]:  # Use top 3 contexts
            text = context['text'][:500]  # Limit length
            
            # Avoid duplicate content
            content_hash = hash(text[:100])
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            page = context['metadata'].get('page', 'Unknown')
            answer_parts.append(f"From page {page}: {text}")
        
        if answer_parts:
            answer = "\n\n".join(answer_parts)
            return answer[:1500] + "..." if len(answer) > 1500 else answer
        else:
            return "I found some information but couldn't extract a clear answer. Please try rephrasing your question."
            
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "I encountered an error while generating the answer. Please try again."

# Routes
@app.route('/status', methods=['GET'])
def status():
    """Return system status"""
    if initialization_error:
        return jsonify({'status': f'Error: {initialization_error}'})
    elif system_ready:
        return jsonify({'status': 'Ready'})
    elif sentence_model is None:
        return jsonify({'status': 'Loading models...'})
    elif index is None:
        return jsonify({'status': 'Building index...'})
    else:
        return jsonify({'status': 'Initializing...'})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'ready': system_ready})

@app.route('/', methods=['GET'])
def index_route():
    """Serve the chatbot UI"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/ask', methods=['POST'])
def ask():
    """Handle chat queries"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        logger.info(f"Query received: {query[:100]}...")
        
        # Check if system is ready
        if not system_ready:
            return jsonify({'answer': 'System is still initializing. Please wait a moment and try again.'})
        
        # Retrieve relevant contexts
        contexts = retrieve(query)
        
        # Generate answer
        answer = generate_answer(query, contexts)
        
        logger.info("Answer generated successfully")
        return jsonify({'answer': answer})
        
    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        return jsonify({'error': 'Internal server error. Please try again.'}), 500

if __name__ == '__main__':
    logger.info("Starting OASIS-E1 Chatbot...")
    
    # Initialize models in background thread
    def init_in_background():
        logger.info("Background initialization started...")
        initialize_models()
    
    thread = threading.Thread(target=init_in_background)
    thread.daemon = True
    thread.start()
    
    # Start Flask app
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
