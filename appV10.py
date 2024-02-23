import streamlit as st
from dotenv import load_dotenv 
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
import os
import requests
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
import time

# Load environment variables from .env file
load_dotenv()

PINECONE_ENVIRONMENT="us-west1-gcp"
pinecone_environment = PINECONE_ENVIRONMENT
INFINEON_LOGO_PATH = 'https://i.ibb.co/jznsjfT/pngwing-com.png'
cloudVectorIndex = 'infidemo2'


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["PINECONE_ENVIRONMENT"] = st.secrets["PINECONE_ENVIRONMENT"]

#openai_api_key = os.getenv("OPENAI_API_KEY")
#pinecone_api_key = os.getenv("PINECONE_API_KEY")

def get_pdf_text(pdf_docs):
    return "".join([page.extract_text() for pdf in pdf_docs for page in PdfReader(pdf).pages])

def get_text_chunks(text):
    return CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len).split_text(text)

def get_vectorstore(text_chunks):
    return FAISS.from_texts(texts=text_chunks, embedding=OpenAIEmbeddings(model="text-embedding-3-large"))

def get_conversation_chain(vectorstore):
    return ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(model="gpt-4-turbo-preview"), retriever=vectorstore.as_retriever(), memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True))

def submit():
    st.session_state.something = st.session_state.widget
    st.session_state.widget = ''

def query_openai_chat_api(prompt, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-4-turbo-preview",  # Adjust model here if needed
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()  # Parse the JSON response

def get_answer(query, k=10, score=False): #For vector cloud based answer recovery
    llm = ChatOpenAI(model_name="gpt-4-0125-preview")
    chain = load_qa_chain(llm, chain_type="stuff")
    similar_docs = st.session_state.cloudVectorIndex.similarity_search_with_score(query, k=k) if score else st.session_state.cloudVectorIndex.similarity_search(query, k=k)
    return chain.run(input_documents=similar_docs, question=query)

def ensure_cloud_vector_index_initialized():
    if "cloudVectorIndex" not in st.session_state or st.session_state.cloudVectorIndex is None:
        try:
            pinecone.init(
                api_key = os.environ["PINECONE_API_KEY"],  
                environment = os.environ["PINECONE_ENVIRONMENT"]  # next to api key in console
            )
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            st.session_state.cloudVectorIndex = Pinecone.from_existing_index(index_name=cloudVectorIndex, embedding=embeddings)
            st.sidebar.success("Cloud Vector Database connected successfully")
        except Exception as e:
            st.sidebar.error(f"Failed to initialize Cloud Vector Database: {e}")

def handle_userinput_cloud_vector_database(user_question):
    ensure_cloud_vector_index_initialized()
    start_time = time.time()
    query = user_question
    response = get_answer(query)
    end_time = time.time()
    response_time = end_time - start_time
  
    if "conversation" in st.session_state:
        # Extract the assistant's response from the chat_response
        assistant_response = response
        assistant_response = f"{response} (Response time: {response_time:.2f} seconds)"
    
        # Update the chat history with the new user question and assistant response
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Display the messages
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            template = bot_template if i % 2 == 0 else user_template
            st.write(template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
    else:
        # Prompt the user to upload documents before asking questions
        st.write("Unable to connect to the vector cloud database")

def handle_userinput_vector_database(user_question):
    # Check if the conversation chain is initialized
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        # Display the messages
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            template = bot_template if i % 2 == 0 else user_template
            st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        # Prompt the user to upload documents before asking questions
        st.write("Please upload documents to process before asking questions.")

def handle_userinput_raw_text(user_question):
    assistant_instructions = "You are a helpful assistant who will receive context for each question. Use the information in the context to answer the question only if it makes sense. Let's approach each query step-by-step, explaining the reasoning behind your answers."
    prompt = f"{assistant_instructions}\n\n{st.session_state.pdf_text}\n\nUser query: {user_question}\n\nAnswer:"
    chat_response = query_openai_chat_api(prompt, os.environ["OPENAI_API_KEY"])

    # Check if the conversation chain is initialized
    if "conversation" in st.session_state:
        # Extract the assistant's response from the chat_response
        assistant_response = chat_response['choices'][0]['message']['content']
    
        # Update the chat history with the new user question and assistant response
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Display the messages
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            template = bot_template if i % 2 == 0 else user_template
            st.write(template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
    else:
        # Prompt the user to upload documents before asking questions
        st.write("Please upload documents to process before asking questions.")


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with AI FAE/SAE", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = None
    if "processing_option" not in st.session_state:
        st.session_state.processing_option = "Cloud Vector Database"
    if "cloudVectorIndex" not in st.session_state:
        st.session_state.cloudVectorIndex = None
    if "enhance_prompts" not in st.session_state:
        st.session_state.enhance_prompts = True
    if 'something' not in st.session_state:
        st.session_state.something = ''


    st.header("Chat with AI FAE/SAE :books:")
    st.text_input("Ask a question:", key='widget', on_change=submit)
    user_question = st.session_state.something

    pdf_text = None  # Initialize pdf_text here
  
    if user_question:
        

        if st.session_state.enhance_prompts:
            assistant_instructions = "Your task is to enhance the user's original query to maximize the relevance and effectiveness of returns from the cloud-based vector database containing specs of Infineon's PSoC 6 MCU. You must only return the query and no other detail of your working. Keep the query length medium."
            prompt = f"{assistant_instructions} User query: {user_question}\n\nAnswer:"
            enhanced_user_question = query_openai_chat_api(prompt, os.environ["OPENAI_API_KEY"])
            enhanced_user_question = enhanced_user_question['choices'][0]['message']['content']
            user_question = enhanced_user_question
            st.markdown(f"<div style='background-color:#2a4a3c; color:white; padding:10px; border-radius:10px;'>{user_question}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color:#2a4a3c; color:white; padding:10px; border-radius:10px;'>{user_question}</div>", unsafe_allow_html=True)

    

        if st.session_state.processing_option == "Process Raw Text":
            handle_userinput_raw_text(user_question)
        if st.session_state.processing_option == "Local Vector Database":
            handle_userinput_vector_database(user_question)
        if st.session_state.processing_option == "Cloud Vector Database":
            handle_userinput_cloud_vector_database(user_question)

        st.session_state.input = '' 
          

    with st.sidebar:
        col1, col2, col3 = st.columns(3)
        with col2:
            st.image(INFINEON_LOGO_PATH,"")
        st.subheader("Import custom knowledge")
        # User option for processing method
        st.session_state.processing_option = st.radio("Choose knowledge base:", ("Cloud Vector Database", "Local Vector Database", "Process Raw Text"))
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type=['pdf'])
        if st.button("Process"):
            try:
                if st.session_state.processing_option == "Local Vector Database":
                    if pdf_docs:  # Check if any documents have been uploaded
                        with st.spinner("Processing"):
                            st.session_state.pdf_text = get_pdf_text(pdf_docs)  # Update pdf_text here
                            st.session_state.conversation = get_conversation_chain(get_vectorstore(get_text_chunks(st.session_state.pdf_text)))
                        st.success("Processing was successful for Local Vector Database method.")
                    else:
                        st.error("No documents uploaded. Please upload documents before processing.")
                elif st.session_state.processing_option == "Process Raw Text":
                    st.session_state.pdf_text = get_pdf_text(pdf_docs)  # Reset pdf_text for raw text processing
                    st.success("Processing was successful for Raw Text method.")
                elif st.session_state.processing_option == "Cloud Vector Database":
                    # initialize pinecone
                    pinecone.init(
                        api_key=os.environ["PINECONE_API_KEY"],  
                        environment=os.environ["PINECONE_ENVIRONMENT"]  
                    )
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
                    st.session_state.cloudVectorIndex = Pinecone.from_existing_index(index_name=cloudVectorIndex, embedding=embeddings)
                    st.success("Processing was successful for Cloud Vector Database method: "+ cloudVectorIndex)
                else:
                    st.error("Please select a processing method.")
            except Exception as e:
                st.error(f"Processing was unsuccessful: {e}")

        st.session_state.enhance_prompts = st.checkbox("Enhance my prompts", value=False, key="enhance_prompts_sidebar")
        # Add credits at the bottom of the page
    st.markdown("""
        <style>
        .footer {
            font-size: 12px;
            color: white;
            text-align: center;
            position: fixed;
            bottom: 0;
            width: 100%;
        }
        </style>
        <div class="footer">
            Developed by Dr N Satheesh Kumar
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
