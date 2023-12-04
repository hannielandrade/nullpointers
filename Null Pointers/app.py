from flask import Flask, render_template, request
from model import predict_sentiment

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])

def result():
    keyword = request.form['keyword']
    sentiment = predict_sentiment(keyword)
    return render_template('result.html', keyword=keyword, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
