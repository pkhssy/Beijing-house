#preprocessing
import pandas as pd
from sklearn.linear_model import LinearRegression

def clean_value(D,str):
    floor = D[str].str.extract('(\d+)')
    floor.columns = [str + '_new']
    D = drop(D,str) #drop original floor data
    D = concat(D,floor) #concat clean_floor to dataset
    return D

def one_hot_data(D,c_str):
    D[c_str] = D[c_str].astype(str)
    one_hot_data = pd.get_dummies(D[[c_str]])
    D = drop(D,c_str)
    new = concat(D,one_hot_data)
    return new

def drop(D,str): return D.drop(columns=[str])
def concat(D1,D2): return pd.concat([D1,D2], axis=1)
def dropnan(D): return D.dropna()

#get data from csv
data = pd.read_csv("./csv/dataset.csv")

print(data.info())

#drop not necessary data
data = drop(data,'tradeTime')
#data = drop(data,'totalPrice')
data = drop(data,'price')
#data = drop(data,'square')

#clean value
data = clean_value(data,'floor')
data = clean_value(data,'constructionTime')

#drop nan data
data_dnan = dropnan(data)

#one hot data [buildingType,buildingStructure]
data_dnan = one_hot_data(data_dnan,'buildingType')
data_dnan = one_hot_data(data_dnan,'buildingStructure')

#make csv
data_dnan.to_csv("./csv/preprocessed.csv", mode='w', index=False)