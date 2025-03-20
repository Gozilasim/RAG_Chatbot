FROM python:3.11.9-slim

WORKDIR /app

COPY requirements.txt .

# Upgrade pip to the latest version
RUN python3 -m pip install --upgrade pip

RUN pip3 install -r requirements2.txt

COPY . /app

EXPOSE 8501


CMD ["streamlit", "run", "chatbot_AIME.py"]

