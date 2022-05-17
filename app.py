from flask import Flask, render_template, request, jsonify
import json
import pickle
import pandas as pd
import numpy as np
import recommendation

app = Flask(__name__)
model = pickle.load(open('model/knn_model.pkl','rb'))

food = pd.read_csv("model/food.csv")
ratings = pd.read_csv("model/ratings.csv")
ratings2 = pd.read_csv("model/ratings2.csv")
preds = pd.read_csv("model/predicted.csv")
preds2 = pd.read_csv("model/predicted2.csv")

preds.rename(columns = {"Unnamed: 0":"Food_ID"}, inplace=True)
preds.reset_index(drop=True,inplace=True)
preds.set_index('Food_ID', inplace=True)
preds.columns = preds.columns.map(float)

preds2.rename(columns = {"Unnamed: 0":"Food_ID"}, inplace=True)
preds2.reset_index(drop=True,inplace=True)
preds2.set_index('Food_ID', inplace=True)
preds2.columns = preds2.columns.map(float)

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
    food_name =     json.loads(request.data)['name']

    print('food name is:',food_name)
    output = recommendation.knn_food_recommendation(food_name)
    # output = output.to_json()
    print(output)
# #    print(jsonify(output))
    return output.to_json()

@app.route("/recommend_svd_api",methods=["POST"])
def recommend_svd_api():
#    food_name = request.get_json(force=True)
    # x = request.json
#    print('food name is:',food_name)
    order_data = json.loads(request.data)
    print(order_data)
    User_id = order_data["userID"]
    del order_data["userID"]
    order_data = pd.DataFrame(order_data)
    recommendation.svd_new_order(User_id, order_data)
    already_rated, predictions = recommendation.svd_food_recommendation(preds2, '2345.0', food, ratings2, 10)
    # predict = predictions.to_json()
    # print(type(predict))
    # print(predict)
    return predictions.to_json()

if __name__ == '__main__':
    app.run(port=3000, debug=False)
    

# import request
# url = 'http://localhost:5000/recommend_api'
# r = requests.post(url,json={foodName:'chicken biryani'})
# print(r.json)