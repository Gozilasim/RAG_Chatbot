from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from database.databseConnection import Base



#OOP method to define table
class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)
    messages = relationship("Message", back_populates="session")  #Automatic synchronization of the relationship
