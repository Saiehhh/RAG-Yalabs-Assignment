This repository contains the source code for the OASIS-E1 Chatbot, a web-based application designed to assist users with questions related to the OASIS-E1 manual. The chatbot leverages natural language processing (NLP) and a FAISS index to retrieve and generate answers from the provided PDF document.
Features

Interactive UI: A modern React-based interface with real-time messaging, quick question suggestions, and a responsive design.
PDF Processing: Extracts and indexes text from the OASIS-E1 manual PDF for efficient querying.
NLP Capabilities: Uses SentenceTransformers for embedding generation and FAISS for similarity search to find relevant content.
Scope Limitation: Responds only to questions about the OASIS-E1 manual, with clear feedback for unrelated queries.
Background Initialization: Loads models and builds the index in a separate thread to ensure a smooth startup.

Requirements

Python 3.10
Required Python packages (listed in requirements.txt):

flask==2.3.3
flask-cors==4.0.0
pdfplumber==0.10.4
nltk==3.8.1
sentence-transformers==2.7.0
faiss-cpu==1.7.4
torch==2.0.1
transformers==4.35.0
numpy==1.24.3



Setup Instructions

Clone the Repository:
bashgit clone https://github.com/yourusername/oasis-e1-chatbot.git
cd oasis-e1-chatbot

Install Dependencies:
Ensure you have Python 3.10 installed, then run:
bashpip install -r requirements.txt

Prepare the PDF:

Ensure the OASIS-E1 manual PDF (e.g., draft-oasis-e1-manual-04-28-2024.pdf) is in the root directory.
Supported filenames: draft-oasis-e1-manual-04-28-2024.pdf, oasis-e1-manual.pdf, or manual.pdf.


Run the Application:

Start the Flask app:
bashpython app.py

Access the chatbot at http://0.0.0.0:7860 in your browser.



Deployment
This project can be deployed on platforms like Hugging Face Spaces. Follow these steps:

Create a new Space on Hugging Face.
Upload all files from this repository, including the PDF.
Use the provided Dockerfile for containerization.
Monitor the "Logs" tab and adjust settings (e.g., hardware tier) if needed.

Usage

Interact with the Chatbot: Open the app in your browser and type questions about the OASIS-E1 manual (e.g., "What is OASIS-E1?").
Quick Questions: Use the suggested questions to get started.
Clear Chat: Click "Clear Chat" to reset the conversation.

Troubleshooting

Internal Server Error: Check local logs or, if deployed, the platform's logs for detailed error messages. Common issues include missing PDF files or memory limits.
No PDF Found: Ensure the PDF is uploaded and matches one of the supported filenames.
Memory Issues: If indexing fails during deployment, consider a higher hardware tier or reducing the PDF processing load.
Initialization Delays: Wait a few moments after startup, as the app initializes models in the background.

License
This project is for educational and demonstration purposes. No specific license is applied, but feel free to modify and use it as needed.
Contact
For issues or questions, open an issue on this GitHub repository or recreate the environment locally for debugging.
Last updated: September 15, 2025, 07:20 PM IST
