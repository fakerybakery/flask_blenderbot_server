print('[Importing modules]')
from flask import Flask, request
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
print('[Starting pipeline]')
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
app = Flask(__name__)
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
@app.route('/')
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