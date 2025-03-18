
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain



DATABASE_URL = "sqlite:///chat_history.db"

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    import Message
    import Session
    Base.metadata.create_all(bind=engine)

#handle database connection
def get_db():
    db = SessionLocal()
    try:
        yield db
        
    finally:
        db.close()

    return db