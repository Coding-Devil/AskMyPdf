import gradio as gr
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.llms import HuggingFaceEndpoint

from pathlib import Path
import chromadb
from unidecode import unidecode

# List of allowed models
allowed_llms = [
    "mistralai/Mistral-7B-Instruct-v0.2", 
    "mistralai/Mixtral-8x7B-Instruct-v0.1", 
    "mistralai/Mistral-7B-Instruct-v0.1",
    "google/gemma-7b-it", 
    "google/gemma-2b-it", 
    "HuggingFaceH4/zephyr-7b-beta", 
    "HuggingFaceH4/zephyr-7b-gemma-v0.1", 
    "meta-llama/Llama-2-7b-chat-hf"
]
list_llm_simple = [os.path.basename(llm) for llm in allowed_llms]

# Load PDF document and create doc splits
def load_doc(list_file_path, chunk_size, chunk_overlap):
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits

# Create vector database
def create_db(splits, collection_name):
    embedding = HuggingFaceEmbeddings()
    new_client = chromadb.EphemeralClient()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        client=new_client,
        collection_name=collection_name,
    )
    return vectordb

# Initialize langchain LLM chain
def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    llm = HuggingFaceEndpoint(
        repo_id=llm_model, 
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_k=top_k,
        load_in_8bit=True,
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    retriever = vector_db.as_retriever()
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff", 
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return qa_chain

# Generate collection name for vector database
def create_collection_name(filepath):
    collection_name = Path(filepath).stem
    collection_name = unidecode(collection_name).replace(" ", "-")
    collection_name = re.sub('[^A-Za-z0-9]+', '-', collection_name)[:50]
    if len(collection_name) < 3:
        collection_name = collection_name + 'xyz'
    if not collection_name[0].isalnum():
        collection_name = 'A' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'Z'
    return collection_name

# Initialize database
def initialize_database(list_file_obj, chunk_size, chunk_overlap, progress=gr.Progress()):
    list_file_path = [x.name for x in list_file_obj if x is not None]
    collection_name = create_collection_name(list_file_path[0])
    doc_splits = load_doc(list_file_path, chunk_size, chunk_overlap)
    vector_db = create_db(doc_splits, collection_name)
    return vector_db, collection_name, "Complete!"

# Initialize LLM
def initialize_LLM(llm_option, llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    llm_name = allowed_llms[llm_option]
    qa_chain = initialize_llmchain(llm_name, llm_temperature, max_tokens, top_k, vector_db, progress)
    return qa_chain, "Complete!"

# Format chat history
def format_chat_history(message, chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history

# Conversation handling
def conversation(qa_chain, message, history):
    formatted_chat_history = format_chat_history(message, history)
    response = qa_chain({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"].split("Helpful Answer:")[-1]
    response_sources = response["source_documents"]
    new_history = history + [(message, response_answer)]
    response_details = [(src.page_content.strip(), src.metadata["page"] + 1) for src in response_sources[:3]]
    return qa_chain, gr.update(value=""), new_history, *sum(response_details, ())

# Gradio Interface
def demo():
    with gr.Blocks(theme="default") as demo:
        vector_db = gr.State()
        qa_chain = gr.State()
        collection_name = gr.State()
        
        gr.Markdown(
        """<center><h2>PDF-based Chatbot</h2></center>
        <h3>Ask any questions about your PDF documents</h3>""")
        
        with gr.Tab("Upload PDF"):
            document = gr.Files(height=100, file_count="multiple", file_types=["pdf"], interactive=True, label="Upload PDF Documents")
        
        with gr.Tab("Process Document"):
            db_btn = gr.Radio(["ChromaDB"], label="Vector Database", value="ChromaDB", type="index")
            with gr.Accordion("Advanced Options", open=False):
                slider_chunk_size = gr.Slider(100, 1000, 600, 20, label="Chunk Size", interactive=True)
                slider_chunk_overlap = gr.Slider(10, 200, 40, 10, label="Chunk Overlap", interactive=True)
            db_progress = gr.Textbox(label="Database Initialization Status", value="None")
            db_btn = gr.Button("Generate Database")
            
        with gr.Tab("Initialize QA Chain"):
            llm_btn = gr.Radio(list_llm_simple, label="LLM Models", value=list_llm_simple[0], type="index")
            with gr.Accordion("Advanced Options", open=False):
                slider_temperature = gr.Slider(0.01, 1.0, 0.7, 0.1, label="Temperature", interactive=True)
                slider_maxtokens = gr.Slider(224, 4096, 1024, 32, label="Max Tokens", interactive=True)
                slider_topk = gr.Slider(1, 10, 3, 1, label="Top-k Samples", interactive=True)
            llm_progress = gr.Textbox(value="None", label="QA Chain Initialization Status")
            qachain_btn = gr.Button("Initialize QA Chain")

        with gr.Tab("Chatbot"):
            chatbot = gr.Chatbot(height=300)
            with gr.Accordion("Document References", open=False):
                for i in range(1, 4):
                    gr.Row([gr.Textbox(label=f"Reference {i}", lines=2, container=True, scale=20), gr.Number(label="Page", scale=1)])
            msg = gr.Textbox(placeholder="Type message here...", container=True)
            gr.Row([gr.Button("Submit"), gr.Button("Clear Conversation")])
            
        # Define Interactions
        db_btn.click(initialize_database, inputs=[document, slider_chunk_size, slider_chunk_overlap], outputs=[vector_db, collection_name, db_progress])
        qachain_btn.click(initialize_LLM, inputs=[llm_btn, slider_temperature, slider_maxtokens, slider_topk, vector_db], outputs=[qa_chain, llm_progress])
        msg.submit(conversation, inputs=[qa_chain, msg, chatbot], outputs=[qa_chain, msg, chatbot] + [None] * 6)

    demo.launch(debug=True)

if __name__ == "__main__":
    demo()
