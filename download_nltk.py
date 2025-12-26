import nltk
import os
import ssl

# --- This part is crucial to bypass potential SSL certificate verification issues ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# --- End of SSL fix ---

# Define the local directory to store NLTK data
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), 'nltk_data')

# Create the directory if it doesn't exist
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)
    print(f"Created directory: {DOWNLOAD_DIR}")

# Download the necessary packages to our local directory
print(f"Downloading NLTK packages to: {DOWNLOAD_DIR}")
nltk.download('punkt', download_dir=DOWNLOAD_DIR)
nltk.download('stopwords', download_dir=DOWNLOAD_DIR)
nltk.download('punkt_tab', download_dir=DOWNLOAD_DIR)

print("\n✅ All necessary NLTK packages have been downloaded successfully.")

# Pre-download sentence-transformer models for faster startup
# These are cached by the library and will be reused at runtime
print("\nPre-downloading ML models for faster startup (this may take a few minutes)...")
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    print("  - Downloading SentenceTransformer model...")
    SentenceTransformer('all-mpnet-base-v2')
    print("  - Downloading CrossEncoder model...")
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("✅ ML models cached successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not pre-download ML models: {e}")
    print("   Models will be downloaded on first request.")

print("\nYou can now run your main application.")