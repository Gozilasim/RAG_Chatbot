from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import asc, delete
from llama_index.llms import ChatMessage
from typing import List

from database.databseConnection import get_db
from database.Session import Session
from database.Message import Message

store = {}

# Function to save a single message
def save_message(session_id: str, role: str, content: str):
    db = next(get_db())
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if not session: # if no same session found, create a new one
            session = Session(session_id=session_id)
            db.add(session)
            db.commit()
            db.refresh(session)
        #else proceed with existing one
        timestamp = datetime.now()  # Get the current timestamp
        db.add(Message(session_id=session.id, role=role, content=content,timestamp=timestamp))
        db.commit()
    except SQLAlchemyError:
        db.rollback()
    finally:
        db.close()

# Function to load chat history
def load_session_history(session_id: str) -> List[ChatMessage]:  # return it as BaseChatMessageHistory object
    db = next(get_db())

    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if session:
            messages = db.query(Message).filter(Message.session_id == session.id).order_by(asc(Message.timestamp)).limit(10).all()

            return[
                {"role": message.role, "content": message.content}
                   for message in messages
        ]

    except SQLAlchemyError:
        pass
    finally:
        db.close()

def get_chat_history(
    chat_messages:List[dict],
) -> List[ChatMessage]:
  
    chat_history = []
    if chat_messages:
        for message in chat_messages:
            chat_history.append(ChatMessage(content=message['content'], role=message['role']))
        return chat_history
    else:
        return []

# Modify the get_session_history function to use the database
def get_session_history(session_id: str) -> List[ChatMessage]:
    chat_history = []
    if session_id not in store:   # commonly for newly open
        messages  = load_session_history(session_id)
        chat_history.extend(get_chat_history(chat_messages=messages))

    return chat_history



# Ensure you save the chat history to the database when needed
def save_all_sessions():
    for session_id, chat_history in store.items():
        for message in chat_history.messages:
            save_message(session_id, message["role"], message["content"])
        delete_all_data(session_id)

# Function to delete all data from the database
def delete_all_data(session_id: str):
    db = next(get_db())
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if not session:
            return
        db.query(Message).filter(Message.session_id == session.id).delete()  # Delete all messages based on sesson id
        db.query(Session).filter(Session.session_id == session.id).delete()  # Delete all sessions  based on sesson id
        db.commit()
    except SQLAlchemyError:
        db.rollback()
    finally:
        db.close()

# Example of saving all sessions before exiting the application
import atexit
atexit.register(save_all_sessions)