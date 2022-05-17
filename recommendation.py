import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle

food = pd.read_csv("model/food.csv")
ratings = pd.read_csv("model/ratings.csv")
ratings2 = pd.read_csv("model/ratings2.csv")
dataset = pd.read_csv("model/users_combined_rating_pivot.csv")
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
    
def knn_food_recommendation(Food_Name):

    dataset.set_index('Food_ID', inplace=True)
    knn_model = pickle.load(open('model/knn_model.pkl','rb'))
    csr_dataset = csr_matrix(dataset.values)
    dataset.reset_index(inplace=True)
    n = 10
    FoodList = food[food['Name'].str.contains(Food_Name)]  
    if len(FoodList):        
        Foodi= FoodList.iloc[0]['Food_ID']
        Foodi = dataset[dataset['Food_ID'] == Foodi].index[0]
        distances , indices = knn_model.kneighbors(csr_dataset[Foodi],n_neighbors=n+1)    
        Food_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])
#         Food_indices = list(zip(indices.squeeze().tolist(),distances.squeeze().tolist()))[:0:-1]
        Recommendations = []
        Given_food_index = Foodi
#        print(Food_indices)
#        print(distances)
        for val in Food_indices:
            Foodi = dataset.iloc[val[0]]['Food_ID']
            i = food[food['Food_ID'] == Foodi].index
            if(i.values==Given_food_index):
                n-=1
                continue
#             print('Food_id:',Foodi,' Name:',food.iloc[i]['Name'].values[0],'with a distance of',val[1])
            Recommendations.append({'Food_id':Foodi,'Name':food.iloc[i]['Name'].values[0],'Distance':val[1]})
        df = pd.DataFrame(Recommendations,index=range(0,n+1))
        return df
    else:
        return "No Similar Foods."

def svd_new_order(User_id, order_data):
#     ratings2.drop(ratings2.tail(1).index,inplace=True)
    uu = pd.read_csv("model/Id.csv")
    uu = uu.astype({"originalId": str})
    fil = uu['originalId'].isin([User_id])
    fil2 = fil.any()
    if(fil2):
        uid = uu.loc[fil]['userId']
        uid = str(float(uid))
    else:
        uid = str(float(uu.tail(1).userId)+1)
        uu.loc[len(uu.index)] = [User_id, uid]
        uu.to_csv("model/Id.csv", index=False)
    for ind in order_data.index:
#         print(rating_data_nested_list['Food_ID'][ind], rating_data_nested_list['rating'][ind])
        foodName = order_data["Food_Name"][ind]
        flt = food['Name'] == foodName
        foodId = int(food.loc[flt]['Food_ID'])
        flt = ratings2['Food_ID'] == foodId
        food_rating = ratings2.loc[flt]['Rating'].mean()
        ratings2.loc[len(ratings2.index)] = [uid, foodId, food_rating]
    
    ratings2.to_csv('model/ratings2.csv', index=False)
    dataset_2 = ratings2.pivot_table(index='User_ID',columns='Food_ID',values='Rating')
    dataset_2.fillna(0,inplace=True)
    R = dataset_2.values
    user_ratings_mean = np.mean(R, axis = 1)
    Ratings_demeaned = R - user_ratings_mean.reshape(-1, 1)
    
    from scipy.sparse.linalg import svds
    U, sigma, Vt = svds(Ratings_demeaned, k = 50)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds2 = pd.DataFrame(all_user_predicted_ratings, columns = dataset_2.columns)
    preds2.to_csv('model/predicted2.csv')
    return


def svd_food_recommendation(predictions, userID, food, original_ratings, num_recommendations):
    
    # Get and sort the user's predictions
    predictions.index = predictions.index.map(str)
    uu = pd.read_csv("model/Id.csv")
    uu = uu.astype({"originalId": str})
    fil = uu['originalId'].isin([userID])
    user_row_number = uu.loc[fil]['userId']
    user_row_number = str(int(float(user_row_number))-1)
    print(str(float(user_row_number)+1))
#     user_row_number = str(int(float(userID))-1)
    select = predictions.loc[predictions.index==user_row_number]
    sorted_user_predictions = select.iloc[0,:].sort_values(ascending=False)
    # Get the user's data and merge in the food information.
    user_data = original_ratings[original_ratings.User_ID.map(str) == str(float(user_row_number)+1)]
    user_full = (user_data.merge(food, how = 'left', left_on = 'Food_ID', right_on = 'Food_ID').
                     sort_values(['Rating'], ascending=False)
                 )
    
    sorted_user_predictions_df = pd.DataFrame(sorted_user_predictions)
    sorted_user_predictions_df.index.name = "Food_ID"
#     sorted_user_predictions_df['Food_ID'] = sorted_user_predictions_df.index
    sorted_user_predictions_df = sorted_user_predictions_df.reset_index()
    # display(user_data)
    # display(sorted_user_predictions_df)
#     print(sorted_user_predictions_df.index)
    
#     Recommend the highest predicted rating foods that the user hasn't tasted yet.
    recommendations = (food[~food['Food_ID'].isin(user_full['Food_ID'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'Food_ID',
               right_on = 'Food_ID').
         rename(columns = {str(user_row_number): 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations


# import json

# # some JSON:
# x =  '{ "userID":"2345", "Food_Name": ["tricolour salad","banana walnut smoothie","andhra pan fried pomfret"]}'

# # parse x:
# order_data = json.loads(x)
# User_id = order_data["userID"]
# del order_data["userID"]
# order_data = pd.DataFrame(order_data)
# order_data

# svd_new_order(User_id, order_data)
# already_rated, predictions = svd_food_recommendation(preds2, '2345.0', food, ratings2, 10)
# predictions

#data = knn_food_recommendation("chicken biryani")
#print(data)
# already_rated, predictions = svd_food_recommendation(preds, 4, food, ratings, 10)
# print(predictions)