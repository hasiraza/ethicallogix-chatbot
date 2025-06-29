from flask import Flask, render_template, request, jsonify
from chatbot import ChatBot
import os

app = Flask(__name__)

# Initialize the chatbot
chatbot = ChatBot()

# Load or train the model
if not chatbot.load_model():
    print("Training new model...")
    chatbot.train()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_bot_response():
    try:
        user_text = request.json.get('message')
        if not user_text:
            return jsonify({'error': 'No message provided'}), 400
        
        response = chatbot.chatbot_response(user_text)
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route("/retrain", methods=["POST"])
def retrain_model():
    """Endpoint to retrain the model with updated intents"""
    try:
        chatbot.train()
        return jsonify({'message': 'Model retrained successfully!'})
    except Exception as e:
        return jsonify({'error': f'Failed to retrain model: {str(e)}'}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("Starting chatbot server...")
    print("Open http://localhost:5000 in your browser to chat!")
    app.run(debug=True, host='0.0.0.0', port=5000)