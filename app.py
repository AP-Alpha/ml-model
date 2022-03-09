import pickle as pkl 
from flask import Flask, render_template, request, url_for, redirect
import numpy as np
import sklearn
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/adm')
def adm():
      return render_template('adm.html')

@app.route('/job')
def index1():
    return render_template('jobs.html')

@app.route('/predict', methods=['POST'])
def predict():
    jee = str(request.form['jee'])
    # team2 = str(request.args.get('list2'))
    print(jee)
    jee = int(jee)

    with open('clg_code.pkl', 'rb') as f:
        clg_code = pkl.load(f)
    with open('college.pkl', 'rb') as f:
        college = pkl.load(f)
    # pkl.load(open('model_sc.pkl', 'rb'))

    with open('model_sc.pkl', 'rb') as f:
        model = pkl.load(f)

    # cteam1 = vocab[team1]
    # cteam2 = vocab[team2]
    # usrip=[]
    # col=['jee']
    # for i in col:
    #     print("==================================================")
    #     usrip.append(eval(input(i+": ")))
    jee=np.array(jee)
    a=jee.reshape(-1,1)
    userpreddt=model.predict(a)
    save=college[clg_code.index(int(userpreddt[0]))]
    print("You may have change to get entrance in: ",save)
    
    # lst = np.array([jee], dtype='int32').reshape(1,-1)

    # prediction = model.predict(lst)

    return render_template('predict.html', data=save)


if __name__ == "__main__":
    app.run(debug=True)


