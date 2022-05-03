import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle

def food_recommendation(Food_Name):
    food = pd.read_csv("model/food.csv")
    dataset = pd.read_csv("model/users_combined_rating_pivot.csv")
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
        print(Food_indices)
        print(distances)
        for val in Food_indices:
            if(val[1]==0.0):
                continue
            Foodi = dataset.iloc[val[0]]['Food_ID']
            i = food[food['Food_ID'] == Foodi].index
            print()
#             print('Food_id:',Foodi,' Name:',food.iloc[i]['Name'].values[0],'with a distance of',val[1])
            Recommendations.append({'Food_id':Foodi,'Name':food.iloc[i]['Name'].values[0],'Distance':val[1]})
        df = pd.DataFrame(Recommendations,index=range(1,n+1))
        return df
    else:
        return "No Similar Foods."