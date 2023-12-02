from flask import Flask,jsonify,request
from model import Chatbot

app = Flask(__name__)
bot = Chatbot()

@app.route("/chat_response",methods=['POST'])
def chat_response():
    prompt = request.json
    print("prompt:",prompt)
    response = bot.generate_responce(prompt['text'],prompt['context'])
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run()