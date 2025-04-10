import os
import streamlit as st
from streamlit_chat import message
from langchain_community.document_loaders import WebBaseLoader
from langchain_writer import WriterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config(
    page_title='ChatBot',
    page_icon=':robot:',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title('ChatBot')

# Initialize session state
if 'chat' not in st.session_state:
    st.session_state['chat'] = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'current_input' not in st.session_state:
    st.session_state.current_input = ''

## function to load and process the website content
def load_website(url):
    try:
        with st.spinner(text="Parsing website...",show_time=False):
            loader = WebBaseLoader(url)
            documents = loader.load()

            print(documents)
            ## split text into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(documents)
            ## create embeddings and store in vector database
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embeddings)

            return vector_store, f"Website parsed successfully"
    except Exception as e:
        return None, f"Error while parsing the website{e}"
    
## create the chat chain
def create_chat_chain(vector_store):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN") 
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        temperature=0.7,
        max_new_tokens=512,
        task="text-generation"
        )
    ## create retreiver
    retreiver = vector_store.as_retriever(search_kwargs={"k": 3})
    ## history aware retreiver
    q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history below, rephrase the user's question to be a standalone question.\n\nChat History: {chat_history}\n\nQuestion: {input}"),
        ("human", "{input}"),
    ])
    retreiver_chain = create_history_aware_retriever(
        llm=llm,
        retriever=retreiver,
        prompt=q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using ONLY the following context:\n\n{context}\n\nIf unsure, say 'I don't know'."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Combine both chains
    conversational_rag_chain = create_retrieval_chain(
        retreiver_chain,
        question_answer_chain
    )
    
    return conversational_rag_chain

def format_chat_history(messages):
    formatted_history = []
    for msg in messages:
        if msg["role"] == "user":
            formatted_history.append(("human", msg["content"]))
        elif msg["role"] == "assistant":
            formatted_history.append(("ai", msg["content"]))
    return formatted_history

# Sidebar
with st.sidebar:
    st.header('🤖')
    link = st.text_input('Paste the website URL below 👇')
    chat = st.button('Chat with me')
    if chat:
        vector_store, message = load_website(link)
        st.session_state.vector_store = vector_store
        st.session_state['chat'] = True
        st.success(message)

# Chat Section
# In your Streamlit app code:

# Chat Section - Updated for continuous conversation
# Chat Section
if st.session_state['chat'] and st.session_state.vector_store is not None:
    st.header('Chat')
    
    # Display chat history
    for i, msg in enumerate(st.session_state.messages):
        message(msg["content"], is_user=(msg["role"] == "user"), key=str(i))
    
    # Input for new message - using empty string as default
    user_input = st.text_input('You:', key='chat_input')
    
    # Use a separate button for submission
    if st.button('Send') and user_input.strip():
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        try:
            # Create chain with current vector store
            chain = create_chat_chain(st.session_state.vector_store)
            
            # Format chat history correctly for the chain
            formatted_history = format_chat_history(st.session_state.messages[:-1])  # Exclude current message
            
            # Get response
            response = chain.invoke({
                "input": user_input,
                "chat_history": formatted_history
            })
            
            bot_response = response["answer"]
            
        except Exception as e:
            bot_response = f"Sorry, I encountered an error: {str(e)}"
        
        # Add bot response to history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        del st.session_state['chat_input']        
        # Force rerun to clear input and update display
        st.rerun()
else:
    if st.session_state['chat'] and st.session_state.vector_store is None:
        st.error("No vector store available. Please try processing the website again.")
    else:
        st.info("Why not try chatting with me? I promise I'll keep it a secret 🤫")