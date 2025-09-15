from flask import Flask, request, jsonify, render_template_string
import pdfplumber
import re
import nltk
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import pickle
import os
from transformers import pipeline
from flask_cors import CORS

# Download NLTK data (run once)
nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)
CORS(app)  # Enable CORS

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
      const [messages, setMessages] = useState([
        {{ '{' }}
          id: 1,
          text: "Hello! I'm your OASIS-E1 assistant. I can help you with questions about the OASIS-E1 manual. What would you like to know?",
          isBot: true,
          timestamp: new Date().toLocaleTimeString([], {{ '{' }}hour: '2-digit', minute:'2-digit'{{ '}' }})
        {{ '}' }}
      ]);
      const [currentMessage, setCurrentMessage] = useState('');
      const [isLoading, setIsLoading] = useState(false);
      const messagesEndRef = useRef(null);
      const inputRef = useRef(null);

      const scrollToBottom = () => {{ '{' }}
        messagesEndRef.current?.scrollIntoView({{ '{' }} behavior: "smooth" {{ '}' }});
      {{ '}' }};

      useEffect(() => {{ '{' }}
        scrollToBottom();
      {{ '}' }}, [messages]);

      const handleSendMessage = async () => {{ '{' }}
        if (!currentMessage.trim() || isLoading) return;

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
        setMessages([
          {{ '{' }}
            id: 1,
            text: "Hello! I'm your OASIS-E1 assistant. I can help you with questions about the OASIS-E1 manual. What would you like to know?",
            isBot: true,
            timestamp: new Date().toLocaleTimeString([], {{ '{' }}hour: '2-digit', minute:'2-digit'{{ '}' }})
          {{ '}' }}
        ]);
      {{ '}' }};

      const quickQuestions = [
        "What is OASIS-E1?",
        "Tell me about discharge planning",
        "How do I assess functional status?",
        "Explain medication management",
        "What are the documentation requirements?"
      ];

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
                  <p className="text-blue-100 text-sm">Your healthcare documentation assistant</p>
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

# Step 1: Parse and Clean PDF
def parse_pdf(pdf_path):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                text = re.sub(r'\nPage \d+\n', '', text)
                text = re.sub(r'OASIS-E1\s+Effective 01/01/2025\s+Centers for Medicare & Medicaid Services\s+\w+', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                sections = re.split(r'\n([A-Z0-9\.\s]{5,})\n', text)
                for i in range(0, len(sections), 2):
                    section_title = sections[i].strip() if i < len(sections) else ''
                    section_text = sections[i+1].strip() if i+1 < len(sections) else sections[i]
                    paras = re.split(r'\n\n', section_text)
                    for para in paras:
                        para = para.strip()
                        if len(para) > 50:
                            chunk = f"{section_title}\n{para}".strip()
                            chunks.append({"text": chunk, "metadata": {"page": page_num, "section": section_title}})
    return chunks

# Step 2: Chunking with overlap
def chunk_with_overlap(chunks, max_tokens=512, overlap=100):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    refined_chunks = []
    for chunk in chunks:
        sentences = nltk.sent_tokenize(chunk['text'])
        current_chunk = []
        current_len = 0
        for sent in sentences:
            sent_len = len(model.encode(sent))
            if current_len + sent_len > max_tokens and current_chunk:
                refined_chunks.append({"text": ' '.join(current_chunk), "metadata": chunk['metadata']})
                current_chunk = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
                current_len = sum(len(model.encode(s)) for s in current_chunk)
            current_chunk.append(sent)
            current_len += sent_len
        if current_chunk:
            refined_chunks.append({"text": ' '.join(current_chunk), "metadata": chunk['metadata']})
    return refined_chunks

# Step 3: Build Index
def build_index(chunks, index_path='faiss_index', embeddings_path='embeddings.pkl'):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [c['text'] for c in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    faiss.write_index(index, index_path)
    with open(embeddings_path, 'wb') as f:
        pickle.dump({'chunks': chunks, 'embeddings': embeddings}, f)
    print(f"Index built with {len(chunks)} chunks!")
    return index, chunks

# Step 4: Retrieve relevant chunks (with relevance threshold)
def retrieve(query, index_path='faiss_index', embeddings_path='embeddings.pkl', top_k=5, relevance_threshold=0.3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
        chunks = data['chunks']
        embeddings = data['embeddings']
    index = faiss.read_index(index_path)
    
    query_emb = model.encode(query)
    _, indices = index.search(np.array([query_emb]).astype('float32'), top_k * 2)
    tokenized_corpus = [doc['text'].lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    candidates = [(i, util.cos_sim(query_emb, embeddings[i])[0][0].item() + bm25_scores[i]) for i in indices[0]]
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in candidates[:top_k]]
    rerank_inputs = [(query, chunks[i]['text']) for i in top_indices]
    rerank_scores = cross_encoder.predict(rerank_inputs)
    sorted_indices = np.argsort(rerank_scores)[::-1]
    selected_chunks = [chunks[top_indices[j]] for j in sorted_indices]
    # Deduplicate and check relevance
    seen = set()
    unique_chunks = []
    max_score = max(rerank_scores) if rerank_scores.size else 0
    if max_score < relevance_threshold:
        return [{"text": "This question is not covered in the OASIS-E1 manual.", "metadata": {"page": 0, "section": "N/A"}}]
    for c in selected_chunks:
        if c['text'] not in seen:
            seen.add(c['text'])
            unique_chunks.append(c)
    return unique_chunks[:top_k]

# Step 5: Generate Answer
def generate_answer(query, contexts):
    generator = pipeline('text-generation', model='distilgpt2')
    if "This question is not covered" in contexts[0]['text']:
        return "This question is not covered in the OASIS-E1 manual."
    context_str = '\n\n'.join([f"From page {c['metadata']['page']}, section {c['metadata']['section']}: {c['text']}" for c in contexts])
    prompt = f"Question: {query}\n\nContext: {context_str}\n\nAnswer the question based only on the context above. Be accurate, concise, and avoid repetition:"
    response = generator(prompt, max_length=300, num_return_sequences=1, truncation=True, no_repeat_ngram_size=3)[0]['generated_text']
    return response.split("Answer the question based only on the context above. Be accurate, concise, and avoid repetition:")[-1].strip()

# Root Route: Serve the UI
@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

# API Endpoint for Queries
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    try:
        if not os.path.exists('faiss_index') or not os.path.exists('embeddings.pkl'):
            print("Building index... This may take 10-20 minutes.")
            raw_chunks = parse_pdf("draft-oasis-e1-manual-04-28-2024.pdf")
            refined_chunks = chunk_with_overlap(raw_chunks)
            build_index(refined_chunks)
        contexts = retrieve(query)
        answer = generate_answer(query, contexts)
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
