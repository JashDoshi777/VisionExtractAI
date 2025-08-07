# VisionExtractAI

VisionExtract: AI Command Center for Document Intelligence
Transforming Static Images into a Dynamic, Conversational Knowledge Base.
VisionExtract is an advanced application designed to redefine the interaction between users and their documents. It moves beyond traditional Optical Character Recognition (OCR) by converting unstructured visual data—such as business cards, brochures, and notes—into a fully interactive and queryable command center. Powered by Google's Gemini Pro and a sophisticated Retrieval-Augmented Generation (RAG) backend, this project demonstrates a next-generation approach to document management and analysis.

Core Concept: Beyond Extraction
The fundamental philosophy of VisionExtract is to extend the utility of extracted data. While conventional OCR tools conclude their process after text extraction, VisionExtract utilizes this information as the foundation for a persistent, intelligent knowledge base.

Each uploaded document contributes to the system's "memory," enabling users to engage in a natural language conversation with their entire document library. This allows for complex queries that can find specific information, synthesize data across multiple files, and even edit contact information directly. The interface is intentionally designed not as a static dashboard, but as a dynamic and responsive environment that reflects the AI's operational process.

Key Features
AI-Powered OCR: Utilizes Google's gemini-pro-vision model for state-of-the-art accuracy in extracting structured contact information from any image, regardless of layout complexity.

Intelligent RAG Chat: Enables natural language conversation with your documents. The AI retrieves relevant context to provide precise answers, grounded exclusively in the user-provided files.

Dual-Mode AI Logic: The chat assistant intelligently discerns user intent, automatically deciding whether to:

Research: Employ the RAG system to find specific answers within unstructured document text.

Synthesize: Use the structured contact data to generate summaries, lists, and tables upon request.

Dynamic and Interactive UI:

Cursor-Reactive Hexagonal Background: A high-performance, animated background of hexagonal tiles that illuminate and react to the user's mouse movements, rendered with HTML5 Canvas.

Glassmorphism Design: A modern, "frosted glass" aesthetic is applied to all UI components, creating a sense of depth and focus.

Animated Card Layout: Extracted contacts are displayed as animated, interactive cards in a responsive grid, replacing traditional static tables.

Intuitive User Experience:

Floating Chat Widget: A clean, accessible chat interface that slides in when needed and remains unobtrusive during data interaction.

Live Activity Feed: A real-time log of the system's actions, from file processing to saving changes, providing transparency to the user.

Inline Editing: Users can click directly on any contact detail on a card to edit it. Changes are saved automatically to the backend.

Powerful Data Management:

Vector Database Backend: Powered by Qdrant, a high-performance vector database that indexes document embeddings for rapid semantic search.

Data Export: Allows for the seamless export of all structured contact data to PDF or Excel (CSV) formats.

Technology Stack & Architecture
This project is built on a modern, AI-first technology stack, selected for high performance and scalability.

Frontend:
HTML5 & CSS3: For the core structure and advanced styling, including custom canvas animations.

Tailwind CSS: For a utility-first methodology to build a clean, responsive user interface.

Vanilla JavaScript (ES6+): For a lightweight, dependency-free, and performant interactive experience.

Backend:
Python & Flask: For a robust and scalable web server and API.

Google Gemini Pro: The core large language model for all vision, extraction, and generative chat tasks.

Qdrant: A high-performance, local-first vector database for indexing document embeddings.

Sentence-Transformers: A lightweight, local model used to generate the vector embeddings that power the RAG system.

Architecture:
The application employs a Retrieval-Augmented Generation (RAG) architecture. When a document is uploaded, its text is extracted, segmented into chunks, and converted into vector embeddings. These embeddings are stored in a Qdrant vector database. When a user submits a query, the system first retrieves the most relevant text chunks from the database and provides them to Gemini Pro as context. This enables the model to generate a highly accurate and factually grounded response.

Setup and Installation
To run VisionExtract on your local machine, please follow these instructions:

Clone the Repository:

git clone https://github.com/your-username/VisionExtract.git
cd VisionExtract

Install Python Dependencies:
Ensure you have Python 3.8 or newer installed. Then, install the required libraries.

pip install Flask Flask-Cors Pillow google-generativeai qdrant-client sentence-transformers

Run the Flask Server:

python app.py

The server will start. The console will display messages indicating the initialization of the RAG system and the loading of the embedding models.

Open the Application:
Launch a web browser and navigate to http://127.0.0.1:5000.

Enter Your API Key:
Provide your Google AI API key to activate the application. It is recommended that billing is enabled on your Google Cloud project to access the higher rate limits of the Gemini Pro model.

Future Development
VisionExtract serves as a robust foundation for further innovation. Potential future enhancements include:

Multi-Modal Chat: Integrating the ability for users to upload images directly within the chat interface and ask questions about them.

Agent-Based Logic: Evolving the backend into an AI agent capable of multi-step reasoning and utilizing an expanded set of tools.

Cloud Deployment: Packaging the application for streamlined deployment to cloud platforms.
