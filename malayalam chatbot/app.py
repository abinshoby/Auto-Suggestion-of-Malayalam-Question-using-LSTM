from flask import Flask
from flask import render_template
from predict_text_meaning import *
from flask import *
app = Flask(__name__)
o=""
@app.route('/')

def index():
    return render_template('index.html',pred=predict)
def sa(v):
    return v
def move_forward():
    #Moving forward code
    #o=predict("ബോട്ടിംഗ")
    index()
    #out=o
    #return (out)

    # template_rendered.__getattribute__("do you mean").set(value=out)


    print("Moving Forward...")
if __name__ == '__main__':
    app.run(debug=True)