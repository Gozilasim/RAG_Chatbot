# copy this command to run -->   streamlit run ".\chatbot_AIME.py"

import streamlit as st
import json
from llama_index import Document, VectorStoreIndex
from fucntions.preprocess_query import validate_input
from database.db_function import save_message, get_session_history,delete_all_data
from llm_model.RAG_pipeline import RAG
from llm_model.config import config
from llm_model.Prompt_Template import fmea_template, general_template,greeting_template, query_wrapper_prompt,question_modifier
from llm_model.DocumentFilter import DocumentFilter

import uuid
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# functions
@st.cache_resource
def initiate_rag():
	return RAG()


def get_session_id():
    if "session_id" not in st.session_state:
        # Generate a new unique session ID
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

session_id = get_session_id()



rag = initiate_rag()

def new_topic_detection(query):
    keywords = ["new topic", "discuss new topic", "start new topic",'how are you', 'hi aime']
    for keyword in keywords:
        if keyword.lower() in query.lower():
            return True
    return False

def set_system_prompt(input_statement,new_topic):
    if "fmea" in input_statement.lower() and not new_topic:
        system_prompt = fmea_template
    elif new_topic:
         system_prompt = greeting_template
    else:
        system_prompt = general_template
    return system_prompt


retrieval = DocumentFilter(headers=rag.headers)


def get_modified_question(query, chat_history, question_llm):
 

    if len(chat_history) > 0:
        # Create a list of Document objects
        documents_chat_history = []
        for chat_message in chat_history:
            document_chat = Document(
                text=chat_message.content,
                metadata={
                    'role': chat_message.role,
                  
                
                }
            )
            documents_chat_history.append(document_chat)

        index_chat_history = VectorStoreIndex.from_documents(documents_chat_history,service_context = rag.service_context, show_progress=False)
        
        question_modifier_engine = index_chat_history.as_query_engine(llm=question_llm)
        latest_question = query + question_modifier
        modified_question = question_modifier_engine.query(latest_question).response

    else:
        modified_question = query
    
    return modified_question 


def getResponse(query, index, return_k):

    new_topic =  new_topic_detection(query)
        # to clear chat history data when move to new topic
    if new_topic:
        print("=======Chat history deleted=======")
        delete_all_data(session_id)
        
    else:
        pass


    # to contextualize the questions
    "____________________________________________________________________________________________________________________________"

    chat_history = get_session_history(session_id)
    print(chat_history)
    print("_______________________________")

    question_llm = rag._init_llm_model(question_modifier, query_wrapper_prompt, endpoint=config['SERVICE_HOST_LLM'], model_name='llama3.1-70b')
    modified_question = get_modified_question(query,chat_history, question_llm)

    print("modified_question: ", modified_question)

    "____________________________________________________________________________________________________________________________"
    
    res_nodes = retrieval.documentRetreival(query = modified_question, index = index, return_k = return_k)

    rerank_nodes = []
 
    for node in res_nodes:
        nodeInfo = json.loads(node.json())['node']
        
        
        newNode = Document(
            id_=nodeInfo['id_'],
            embedding = nodeInfo['embedding'],
            metadata = nodeInfo['metadata'],
            relationships = nodeInfo['relationships'],
            text = nodeInfo['text'],
        )

        rerank_nodes.append(newNode)


    index_2 = VectorStoreIndex(rerank_nodes,service_context = rag.service_context, show_progress=False)


    system_prompt = set_system_prompt(modified_question,new_topic)
    custom_llm = rag._init_llm_model(system_prompt,query_wrapper_prompt, endpoint=config['SERVICE_HOST_LLM'], model_name='llama3.1-70b')
    

    query_engine = index_2.as_chat_engine(llm=custom_llm,
                                        chat_mode="context",
                                        system_prompt =system_prompt )


    response = query_engine.chat(modified_question)

    for message in query_engine.chat_history:
        save_message(session_id, message.role, message.content)

    return response



# Function to read the CSS file and inject it into the Streamlit app
def inject_custom_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def display_message(message):

    if message["role"] == "user":
        div = f"""
            <div class="chat-row user-row row-reverse">
                <img src="app/static/profile.png" width=50 height =50>
                <div class="chat-content user-content">
                    <div>{message["content"]}</div>
                </div>
            <div>
        """
        st.markdown(div, unsafe_allow_html=True)
        

    elif message["role"] == "assistant":
        div = f"""
            <div class="chat-row assistant-row ">
                <img src="app/static/chatbot3.png" width=50 height = 50>
                <div class="chat-content assistant-content">
                    <div>{message["content"]}</div>
                </div>
            <div>
        """
        st.markdown(div, unsafe_allow_html=True)

#GUI
def main():

    #switch to ur path
    inject_custom_css("static/style.css")
    # Custom CSS to set font sizes and reduce spacing
    st.markdown(
        """
        <style>
        .big-font {
            font-size:30px !important;
            margin-bottom: 0px !important;
            padding-bottom: 0px !important;
        }
        .small-font {
            font-size:10px !important;
            margin-top: 0px !important;
            padding-top: 0px !important;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Applying the custom CSS classes to the markdown text
    st.markdown('<p class="big-font">AIME - Infineon AI powered Manufacturing Engineer</p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">( MAL Prototype v1 )</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.write(f"Session ID: {session_id}")
        if st.button("New Chat"):
                # Reset session data and assign a new session ID
                print("=======Chat history deleted (button clicked)=======")
                delete_all_data(session_id)
                st.session_state.clear()
                st.session_state.session_id = get_session_id()
                st.session_state.messages=[]
                st.session_state.greetings = False
                print(f"session ID:  {session_id} ")

    #st.title("Infineon AI powered Manufacturing Engineer - MAL Prototype v1")
    if st.button("Help"):
        st.info("This is a chatbot interface. "
                "You can type your questions in the input box below and get responses. "
                "For more detailed instructions, refer to the user guide."
                )

    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.greetings = False

    if not st.session_state.greetings:
        
        intro = "Hi, I'm AIME, an AI-powered Manufacturing Engineer at Infineon Technologies. I'm here to help answer your questions and provide information on various topics related to IFX BE Assembly & Test processes. What would you like to discuss? Please go ahead and introduce the topic you'd like to explore. I'll do my best to provide detailed and accurate information. In the end, I'll summarize the key points for your reference. Let's get started!"
        
        st.session_state.messages.append({'role': 'assistant', 'content': intro})
        st.session_state.greetings = True
    
    for message in st.session_state.messages:
        display_message(message)


    if prompt := (st.chat_input("Type Your Prompt Here", max_chars= 600)):

        is_valid, error_message = validate_input(prompt)
        if not is_valid:
            st.error(error_message)
        else:
            div = f"""
                <div class="chat-row user-row row-reverse">
                    <img src="app/static/profile.png" width=50 height =50>
                    <div class="chat-content user-content">
                        <div>{prompt}</div>
                    </div>
                </div>
            """
            st.markdown(div, unsafe_allow_html=True)

            st.session_state.messages.append({'role':'user','content':prompt}) 
            with st.spinner('Generating response...'):
                response = getResponse(query = prompt, index = rag.idx, return_k = 50)
             
                  
            st.session_state.messages.append({'role': 'assistant', 'content': response})
            div = f"""
                <div class="chat-row assistant-row ">
                    <img src="app/static/chatbot3.png" width=50 height = 50>
                    <div class="chat-content assistant-content">
                        <div>{response}</div>
                    </div>
                </div>
            """
            st.markdown(div, unsafe_allow_html=True)


if __name__ == "__main__":
    main()





