from flask import Flask
from flask import render_template
import predict_text_meaning
from flask import *
import imp
app = Flask(__name__)
o=""
#@app.route('/')

# def index():
#     return render_template('index.html',pred=predict)
# @app.route('/pred', methods=['POST'])
# def pred():
#     data=request.form['inp'];
#     return predict(data)
@app.route('/')
def first():
    return render_template('predict.html')
@app.route('/req', methods=['POST'])
def req():
    inp =  request.form['inp'];
    if(len(inp)>0):
        out= json.dumps({'status':'OK','suggestion':predict_text_meaning .predict([inp])});#json.dumps({'status':'OK','user':user,'pass':password});

        imp.reload(predict_text_meaning)
        return  out
    else:
        return json.dumps({'status':'OK','suggestion':''});
if __name__ == '__main__':
    app.run(debug=True)