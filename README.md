Aurora - AI Career Coach Chatbot
Aurora is an AI-powered chatbot that provides personalized career coaching by simulating human-like traits such as emotion, memory, and personality. It uses natural language understanding and a local LLM API to deliver meaningful, context-aware advice.

Features
Simulates brain chemistry (dopamine, serotonin, cortisol, energy, cognitive load)

Adapts personality and tone based on mood and emotional state

Extracts and remembers user details across conversations

Uses cognitive biases for more human-like responses

Web-based interface built with Tailwind CSS

Connects to a local LLM (e.g. OpenChat via Ollama API)

Tech Stack
Python (Flask, spaCy, NLTK, requests)

JavaScript, HTML, Tailwind CSS

Flask-CORS

Ollama API (OpenChat model)

Project Structure
aurora-career-coach/
├── app.py - Flask backend
├── static/index.html - Chat UI
├── data/ - Memory and emotional state files (JSON)
├── requirements.txt - Python dependencies
└── README.md - Project overview

Installation
Clone the repository:

git clone https://github.com/YOUR_USERNAME/aurora-career-coach.git
cd aurora-career-coach

Install dependencies:

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader vader_lexicon

Run the app:

python app.py

Then go to http://localhost:5000 in your browser.

Deployment
You can deploy Aurora using:

Render.com (Flask with static files)

Railway.app

Your own server (Gunicorn + Nginx)

License
This project is open for educational and personal use. Feel free to fork and modify.

Contact
Created by Boma Halliday
Email: boma913@gmail.com


