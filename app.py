import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
import docx
import zipfile
import tempfile
import os
import mimetypes
from google.oauth2 import service_account
from PyPDF2 import PdfReader
from io import BytesIO
import tiktoken

# Initialize tiktoken encoding
enc = tiktoken.get_encoding("cl100k_base")

# -------------------- File Reading Functions -------------------- #

def read_docx(file):
    """Read a .docx file and return its text content."""
    doc = docx.Document(file)
    full_text = [para.text for para in doc.paragraphs]
    return '\n'.join(full_text)

def read_txt(file):
    """Read a .txt file and return its text content."""
    text = file.read().decode('utf-8')
    return text

def read_pdf(file):
    """Read a .pdf file and return its text content."""
    reader = PdfReader(file)
    full_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text.append(text)
    return '\n'.join(full_text)

# -------------------- File Processing Functions -------------------- #

def process_uploaded_files(uploaded_files):
    """Process uploaded files and extract text content."""
    document_contents = []
    for uploaded_file in uploaded_files:
        try:
            file_type = uploaded_file.type
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension == '.docx' or file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                # Handle .docx files
                file_content = read_docx(uploaded_file)
                document_contents.append(f"File: {uploaded_file.name}\nContent: {file_content}\n")
            elif file_extension == '.txt' or file_type == "text/plain":
                # Handle .txt files
                file_content = read_txt(uploaded_file)
                document_contents.append(f"File: {uploaded_file.name}\nContent: {file_content}\n")
            elif file_extension == '.pdf' or file_type == "application/pdf":
                # Handle .pdf files
                file_content = read_pdf(uploaded_file)
                document_contents.append(f"File: {uploaded_file.name}\nContent: {file_content}\n")
            elif file_extension == '.zip' or file_type == "application/zip":
                # Handle .zip files
                zip_contents = process_zip_file(uploaded_file)
                document_contents.extend(zip_contents)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Could not read file {uploaded_file.name}: {str(e)}")
    return '\n'.join(document_contents)

def process_zip_file(zip_file):
    """Extract and process text files from a ZIP archive."""
    document_contents = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            with zipfile.ZipFile(zip_file) as z:
                z.extractall(tmpdirname)
                for root, dirs, files in os.walk(tmpdirname):
                    for filename in files:
                        filepath = os.path.join(root, filename)
                        file_extension = os.path.splitext(filename)[1].lower()
                        try:
                            if file_extension == '.docx':
                                with open(filepath, 'rb') as f:
                                    file_content = read_docx(f)
                                    document_contents.append(f"File: {filename}\nContent: {file_content}\n")
                            elif file_extension == '.txt':
                                with open(filepath, 'r', encoding='utf-8') as f:
                                    file_content = f.read()
                                    document_contents.append(f"File: {filename}\nContent: {file_content}\n")
                            elif file_extension == '.pdf':
                                with open(filepath, 'rb') as f:
                                    file_content = read_pdf(f)
                                    document_contents.append(f"File: {filename}\nContent: {file_content}\n")
                            else:
                                st.warning(f"Unsupported file type in zip: {filename}")
                        except Exception as e:
                            st.error(f"Could not read file {filename} in zip: {str(e)}")
        except zipfile.BadZipFile:
            st.error("Invalid ZIP file. Please upload a valid ZIP archive.")
    return document_contents

# -------------------- Utility Functions -------------------- #

def chunk_content(text, max_length=5000):
    """Split text into smaller chunks to avoid exceeding token limits."""
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def summarize_content(content, model):
    """Summarize content that exceeds token limits."""
    st.info("Content is large. Summarizing to fit the model's context window...")
    prompt = f"Summarize the following content:\n\n{content}"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error summarizing content: {str(e)}")
        return ""

def count_tokens_tiktoken(text):
    """Count tokens in text using tiktoken."""
    tokens = enc.encode(text)
    return len(tokens)

def count_tokens(text, chunk_size=10000):
    """Count tokens in text by processing in chunks."""
    total_tokens = 0
    text_length = len(text)
    for i in range(0, text_length, chunk_size):
        chunk = text[i:i+chunk_size]
        tokens = count_tokens_tiktoken(chunk)
        total_tokens += tokens
    return total_tokens

# -------------------- Main Functionality -------------------- #

def ask_gemini(question, context, model):
    """Generate an answer from the Gemini model based on the question and context."""
    chunks = chunk_content(context)
    responses = []
    
    progress_bar = st.progress(0)
    for idx, chunk in enumerate(chunks):
        prompt = f"Context:\n{chunk}\n\nQuestion: {question}\n\nAnswer:"
        try:
            response = model.generate_content(prompt)
            responses.append(response.text)
        except Exception as e:
            st.error(f"Error from Gemini API: {str(e)}")
            return ""  # Stop processing if there's an error
        progress_bar.progress((idx + 1) / len(chunks))
    
    progress_bar.empty()
    
    if len(responses) > 1:
        # Combine and summarize responses
        final_prompt = f"Summarize the following responses to the question: '{question}'\n\n" + "\n\n".join(responses)
        try:
            final_response = model.generate_content(final_prompt)
            return final_response.text
        except Exception as e:
            st.error(f"Error from Gemini API during summarization: {str(e)}")
            return ""
    else:
        return responses[0]

# -------------------- Streamlit App -------------------- #

def main():
    st.title("Document Analysis with Gemini")
    
    # Instructions
    st.write("""
    **Instructions:**
    - Upload your text-based documents (.txt, .docx, .pdf) or a ZIP file containing these documents.
    - Enter a question related to the content of the documents.
    - Adjust the model parameters if necessary.
    - Click 'Get Answer' to receive a response from the Gemini model.
    """)
    
    # Load service account credentials from secrets
    credentials = service_account.Credentials.from_service_account_info(st.secrets["service_account"])
    
    # Initialize Vertex AI with credentials
    vertexai.init(project="your-project-id", location="us-central1", credentials=credentials)
    
    # Model selection
    model_options = ["gemini-1.5-flash-001", "gemini-1.5-pro-001"]
    selected_model = st.selectbox("Select Gemini Model:", model_options)
    
    # Parameter adjustments
    st.sidebar.header("Model Parameters")
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7)
    max_output_tokens = st.sidebar.slider("Max Output Tokens:", min_value=1, max_value=8192, value=256)
    
    # Set MAX_TOTAL_TOKENS based on selected model
    if selected_model == "gemini-1.5-flash-001":
        MAX_TOTAL_TOKENS = 950_000
    elif selected_model == "gemini-1.5-pro-001":
        MAX_TOTAL_TOKENS = 1_950_000
    else:
        MAX_TOTAL_TOKENS = 8000  # default
    
    # Create the model with parameters
    model = GenerativeModel(selected_model).with_parameters(temperature=temperature, max_output_tokens=max_output_tokens)
    
    # File uploader to upload multiple files, including ZIP files
    uploaded_files = st.file_uploader(
        "Upload .txt, .docx, .pdf, or .zip files",
        type=["txt", "docx", "pdf", "zip"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Create a hashable representation of the uploaded files
        files_info = tuple((f.name, f.size) for f in uploaded_files)
        if 'document_contents' not in st.session_state or st.session_state.get('files_hash') != hash(files_info):
            with st.spinner("Processing uploaded files..."):
                # Process the uploaded files
                full_document_content = process_uploaded_files(uploaded_files)
                if full_document_content.strip() == "":
                    st.error("No valid text content found in the uploaded files.")
                    return
                # Use tiktoken to count tokens
                context_tokens = count_tokens(full_document_content)
                if context_tokens > MAX_TOTAL_TOKENS:
                    full_document_content = summarize_content(full_document_content, model)
                # Cache the content and files hash
                st.session_state['document_contents'] = full_document_content
                st.session_state['files_hash'] = hash(files_info)
                st.success("Documents loaded successfully!")
        else:
            full_document_content = st.session_state['document_contents']
    else:
        st.info("Please upload files to proceed.")
        return
    
    # User input
    user_question = st.text_input("Ask a question about the uploaded documents:")
    
    # Token counting button
    if st.button("Count Tokens"):
        if user_question:
            try:
                question_tokens = count_tokens(user_question)
                context_tokens = count_tokens(full_document_content)
                st.write(f"**Question Token Count:** {question_tokens}")
                st.write(f"**Context Token Count:** {context_tokens}")
                st.write(f"**Total Token Count:** {question_tokens + context_tokens}")
            except Exception as e:
                st.error(f"Error counting tokens: {str(e)}")
        else:
            st.warning("Please enter a question.")
    
    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Generating answer... This may take a while."):
                try:
                    answer = ask_gemini(user_question, full_document_content, model)
                    if answer:
                        st.subheader("Answer:")
                        st.write(answer)
                except Exception as e:
                    st.error(f"Error processing request: {str(e)}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
