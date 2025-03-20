from sqlalchemy import Column, Integer, String, Text, ForeignKey,DateTime
from sqlalchemy.orm import relationship
from database.databseConnection import Base
from datetime import datetime


#OOP method to define table
class Message(Base):
    __tablename__ = "messages"
    message_id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    session = relationship("Session", back_populates="messages")  #Automatic synchronization of the relationship