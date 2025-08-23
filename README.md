# VisionExtractAI

VisionExtractAI is a powerful Python application designed to automate the extraction of contact information from business cards and multi-page brochures. It leverages Google's Gemini API for high-accuracy Optical Character Recognition (OCR) and features an integrated AI chatbot for querying the extracted data.

Overview
This tool provides a streamlined workflow for digitizing physical documents. Users can upload image files (for business cards) or PDFs (for brochures), and the application will automatically parse the content, structure it into a clean JSON format, and save it for future use. The key feature is a Retrieval-Augmented Generation (RAG) system that allows users to have natural language conversations with their extracted data.

Technology Stack
The application is built with a modern, efficient technology stack:

AI & Machine Learning:

Google Gemini 1.5 Pro & Flash: Utilized for state-of-the-art OCR and data structuring.

spaCy: Powers the Natural Language Processing (NLP) components of the RAG system.

Backend:

Python: The core programming language for the application.

Flask: A lightweight web framework used to serve the backend API and user interface.

Frontend:

HTML, CSS, JavaScript: Standard web technologies for the user interface.

Key Features
Multi-Format Support: Accepts common image formats (.jpg, .png) for business cards and PDFs for brochures.

High-Accuracy Extraction: Employs advanced AI models to accurately extract names, email addresses, and phone numbers, even from complex layouts.

Structured Data Output: All extracted information is cleaned and formatted into a structured JSON object.

Conversational AI Chat (RAG): Users can ask questions in natural language (e.g., "List all contacts from PartnerRe") to instantly retrieve information from the extracted data.

Persistent Storage: Saves all extracted contacts, creating a searchable and permanent digital database.
