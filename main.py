print('[Importing modules]')
from flask import Flask, request
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from flask_cors import CORS, cross_origin
print('[Starting pipeline]')
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
@app.route('/', methods=['GET', 'POST'])
def api():
    if request.method == 'POST':
        formdata = request.form
    else:
        formdata = request.args
    message = formdata['message']
    inputs = tokenizer(message, return_tensors="pt")
    result = model.generate(**inputs)
    message_bot = tokenizer.decode(result[0], skip_special_tokens=True)
    return message_bot
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
