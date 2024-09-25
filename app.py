import streamlit as st
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
import docx
import zipfile
import tempfile
import os
import mimetypes
from google.oauth2 import service_account
from PyPDF2 import PdfReader
from io import BytesIO
import tiktoken
from concurrent.futures import ThreadPoolExecutor

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

def chunk_content_by_tokens(text, max_tokens):
    """Split text into smaller chunks based on token count."""
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end
    return chunks

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

def chunk_and_summarize_content(content, question, model, generation_config, selected_model):
    """Chunk and summarize content that exceeds token limits."""
    st.info("Content is large. Summarizing in chunks to focus on the question...")
    # Set chunk size and rate limits based on model
    if selected_model == "gemini-1.5-flash-002":
        MAX_CONTEXT_TOKENS = 950_000
        REQUESTS_PER_MINUTE = 200
        TOKENS_PER_MINUTE = 4_000_000
    elif selected_model == "gemini-1.5-pro-002":
        MAX_CONTEXT_TOKENS = 1_950_000
        REQUESTS_PER_MINUTE = 60
        TOKENS_PER_MINUTE = 4_000_000
    else:
        MAX_CONTEXT_TOKENS = 8000
        REQUESTS_PER_MINUTE = 60
        TOKENS_PER_MINUTE = 200_000

    overhead_tokens = 2000  # Reserve tokens for prompt and overhead
    max_tokens_per_chunk = MAX_CONTEXT_TOKENS - overhead_tokens
    # We may choose to set a smaller chunk size to avoid hitting rate limits
    chunk_size_tokens = min(100_000, max_tokens_per_chunk)

    chunks = chunk_content_by_tokens(content, chunk_size_tokens)

    # Calculate max workers based on rate limits
    max_workers = min(REQUESTS_PER_MINUTE // 60, 10)  # Limit to 10 workers
    summaries = []
    progress_bar = st.progress(0)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, chunk in enumerate(chunks):
            prompt = f"""
You are an expert assistant. Based on the following content, extract information that is relevant to the question: '{question}'.

Content:
{chunk}

Provide a concise summary that focuses only on the parts of the content that are most relevant to the question.
"""
            future = executor.submit(model.generate_content, prompt, generation_config=generation_config)
            futures.append((idx, future))

        for idx, future in enumerate(futures):
            try:
                response = future[1].result()
                summaries.append(response)
            except Exception as e:
                st.error(f"Error summarizing chunk {idx+1}: {str(e)}")
                return None
            progress_bar.progress((idx + 1) / len(chunks))

    progress_bar.empty()

    # Combine summaries
    combined_summary = "\n".join([resp.text for resp in summaries])
    return combined_summary

def ask_gemini(question, context, model, generation_config, selected_model, system_prompt=""):
    """Generate an answer from the Gemini model based on the question and context."""
    # Set chunk size based on model
    if selected_model == "gemini-1.5-flash-002":
        MAX_CONTEXT_TOKENS = 950_000
    elif selected_model == "gemini-1.5-pro-002":
        MAX_CONTEXT_TOKENS = 1_950_000
    else:
        MAX_CONTEXT_TOKENS = 8000

    overhead_tokens = 2000  # Reserve tokens for prompt and overhead
    max_tokens_per_chunk = MAX_CONTEXT_TOKENS - overhead_tokens
    # We may choose to set a smaller chunk size to avoid hitting rate limits
    chunk_size_tokens = min(100_000, max_tokens_per_chunk)

    chunks = chunk_content_by_tokens(context, chunk_size_tokens)
    responses = []

    progress_bar = st.progress(0)

    # Process chunks concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for idx, chunk in enumerate(chunks):
            if system_prompt:
                prompt = f"{system_prompt}\n\nContext:\n{chunk}\n\nQuestion: {question}\n\nAnswer:"
            else:
                prompt = f"Context:\n{chunk}\n\nQuestion: {question}\n\nAnswer:"
            future = executor.submit(model.generate_content, prompt, generation_config=generation_config)
            futures.append((idx, future))

        for idx, future in enumerate(futures):
            try:
                response = future[1].result()
                responses.append(response)
            except Exception as e:
                st.error(f"Error from Gemini API: {str(e)}")
                return ""  # Stop processing if there's an error
            progress_bar.progress((idx + 1) / len(chunks))

    progress_bar.empty()

    if len(responses) > 1:
        # Combine and summarize responses
        responses_text = "\n\n".join([resp.text for resp in responses])
        final_prompt = f"Based on the following responses, provide a concise and accurate answer to the question: '{question}'\n\n{responses_text}"
        if system_prompt:
            final_prompt = f"{system_prompt}\n\n{final_prompt}"
        try:
            final_response = model.generate_content(final_prompt, generation_config=generation_config)
            return final_response.text
        except Exception as e:
            st.error(f"Error from Gemini API during summarization: {str(e)}")
            return ""
    else:
        return responses[0].text

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
    - You can adjust settings on the left-hand side, but they are set up to work well for most use cases.
    - Please note that the model is not perfect and may not always provide accurate answers.
    - Please be patient as the model may take a few minutes to respond.
    - You can modify the system prompt to change the behavior of the model by clicking the checkbox after file upload.
    """)

    # Load service account credentials from secrets
    credentials = service_account.Credentials.from_service_account_info(st.secrets["service_account"])

    # Initialize Vertex AI with credentials
    vertexai.init(project="vertex-ai-development", location="us-central1", credentials=credentials)

    # Model selection
    model_options = ["gemini-1.5-flash-002", "gemini-1.5-pro-002"]
    selected_model = st.selectbox("Select Gemini Model:", model_options)

    # Parameter adjustments
    st.sidebar.header("Model Parameters")
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7)
    max_output_tokens = st.sidebar.slider("Max Output Tokens:", min_value=1, max_value=8192, value=8192)
    top_p = st.sidebar.slider("Top P:", min_value=0.0, max_value=1.0, value=1.0)
    top_k = st.sidebar.slider("Top K:", min_value=1, max_value=40, value=40)

    # Set MAX_TOTAL_TOKENS based on selected model
    if selected_model == "gemini-1.5-flash-002":
        MAX_TOTAL_TOKENS = 950_000
    elif selected_model == "gemini-1.5-pro-002":
        MAX_TOTAL_TOKENS = 1_950_000
    else:
        MAX_TOTAL_TOKENS = 8000  # default

    # Create the model
    model = GenerativeModel(selected_model)

    # Create the generation config
    generation_config = GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        top_k=top_k
    )

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

    # Custom system prompt
    use_custom_system_prompt = st.checkbox("Use custom system prompt")
    if use_custom_system_prompt:
        system_prompt = st.text_area("Enter system prompt:")
    else:
        system_prompt = ""  # You can set a default system prompt here if desired

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
            # Retrieve full_document_content from session state
            full_document_content = st.session_state.get('document_contents', '')
            if not full_document_content:
                st.error("No document content available.")
                return

            # Compute total tokens
            question_tokens = count_tokens(user_question)
            context_tokens = count_tokens(full_document_content)
            total_tokens = question_tokens + context_tokens

            if total_tokens > MAX_TOTAL_TOKENS:
                st.warning("The total content exceeds the model's maximum context size. The content will need to be summarized, which may reduce granularity and affect accuracy.")
                with st.spinner("Summarizing the content... This may take a while."):
                    summarized_content = chunk_and_summarize_content(full_document_content, user_question, model, generation_config, selected_model)
                    if summarized_content is None:
                        st.error("Error during summarization.")
                        return
                    full_document_content = summarized_content
                    # Update the content in session state
                    st.session_state['document_contents'] = full_document_content
            with st.spinner("Generating answer... This may take a while."):
                try:
                    answer = ask_gemini(user_question, full_document_content, model, generation_config, selected_model, system_prompt)
                    if answer:
                        st.subheader("Answer:")
                        st.write(answer)
                except Exception as e:
                    st.error(f"Error processing request: {str(e)}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
