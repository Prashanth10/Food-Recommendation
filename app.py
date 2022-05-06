from flask import Flask, render_template, request, jsonify
import json
import pickle
import pandas as pd
import numpy as np
import recommendation

app = Flask(__name__)
model = pickle.load(open('model/knn_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('main.html')
#   return 'Main page'

@app.route("/recommend_knn", methods=["POST"])
def recommend_knn():
    food_name = request.form['foodName']
    print('food name is:',food_name)
    output = recommendation.knn_food_recommendation(food_name)
    return render_template('main.html', recommendation_text = 'Recommendations are $ {}'.format(output))

@app.route("/recommend_knn_api",methods=["POST"])
def recommend_knn_api():
#    food_name = request.get_json(force=True)
    food_name = request.form['foodName']
#    print('food name is:',food_name)
    output = recommendation.knn_food_recommendation(food_name)
    output = output.to_json()
    # json.loads(output)
#    print(jsonify(output))
    return output

if __name__ == '__main__':
    app.run(port=3000, debug=False)
    

# import request
# url = 'http://localhost:5000/recommend_api'
# r = requests.post(url,json={foodName:'chicken biryani'})
# print(r.json)