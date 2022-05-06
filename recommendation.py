import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle

food = pd.read_csv("model/food.csv")
ratings = pd.read_csv("model/ratings.csv")
dataset = pd.read_csv("model/users_combined_rating_pivot.csv")
preds = pd.read_csv("model/predicted.csv")

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


def svd_food_recommendation(predictions, userID, food, original_ratings, num_recommendations):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # User ID starts at 1, not 0
    sorted_user_predictions = preds.iloc[user_row_number].sort_values(ascending=False) # User ID starts at 1
    
    # Get the user's data and merge in the food information.
    user_data = original_ratings[original_ratings.User_ID == (userID)]
    user_full = (user_data.merge(food, how = 'left', left_on = 'Food_ID', right_on = 'Food_ID').
                     sort_values(['Rating'], ascending=False)
                 )

    # print('User {0} has already rated {1} food.'.format(userID, user_full.shape[0]))
    # print('Recommending highest {0} predicted ratings food not already rated.'.format(num_recommendations))
    # display(user_full)
    # display(pd.DataFrame(sorted_user_predictions).reset_index())
    
     # Recommend the highest predicted rating foods that the user hasn't tasted yet.
    recommendations = (food[~food['Food_ID'].isin(user_full['Food_ID'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'Food_ID',
               right_on = 'Food_ID').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations

#data = knn_food_recommendation("chicken biryani")
#print(data)
# already_rated, predictions = svd_food_recommendation(preds, 4, food, ratings, 10)
# print(predictions)