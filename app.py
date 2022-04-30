from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('\model\knn_model.pkl','rb'))
app = Flask(__name__)

app.route('/')
def home():
#     return render_template('main.html')
    return 'Main page'

app.route("/recommend",methods=["POST"])
def recommend():
    food_name = request.form['foodName']
    return render_template('main.html', recommendation_text = 'Recommendations are $ {}'.format(output))

app.route("/recommend_api",methods=["POST"])
def recommend_api():
    food_name = request.get_json(force=True)
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
    

# import request
# url = 'http://localhost:5000/recommend_api'
# r = requests.post(url,json={foodName:'chicken biryani'})
# print(r.json)