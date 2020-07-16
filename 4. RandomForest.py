#train
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def drop(D,str): return D.drop(columns=[str])
def concat(D1,D2): return pd.concat([D1,D2], axis=1)
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

data = pd.read_csv("./csv/preprocessed.csv")

train_set, test_set = split_train_test(data, 0.2)
print (len(train_set), "train +", len(test_set), "test")
train_set_x, test_set_x = drop(train_set,'totalPrice'), drop(test_set,'totalPrice')
train_set_y, test_set_y = train_set['totalPrice'], test_set['totalPrice']

def find_estimators(train_set_x,train_set_y,test_set_x,test_set_y):
    #Random forest training in the size range of the ensemble to check for changes in MSE
    mseOos = []
    nTreeList = range(10, 500, 100)
    for iTrees in nTreeList:
        depth = None
        maxFeat = None
        wineRFModel = RandomForestRegressor(n_estimators=iTrees,max_depth=None, max_features=maxFeat,oob_score=False, random_state=531)
        wineRFModel.fit(train_set_x, train_set_y)
        #MSE accumulation for data sets
        prediction = wineRFModel.predict(test_set_x)
        mseOos.append(mean_squared_error(test_set_y, prediction))
    print("MSE")
    print(mseOos[-1])
    #Draw a diagram of the number of trees in the ensemble against the training test error
    plt.plot(nTreeList, mseOos)
    plt.xlabel('Number of Trees in Ensemble')
    plt.ylabel('Mean Squared Error')
    plt.show()

def find_max_depth(train_set_x,train_set_y,test_set_x,test_set_y):
    #Random forest training in the size range of the ensemble to check for changes in MSE
    mseOos = []
    nTreeList = range(10, 50, 1)
    for iTrees in nTreeList:
        maxFeat = None
        wineRFModel = RandomForestRegressor(n_estimators=30,max_depth=iTrees, max_features=maxFeat,oob_score=False, random_state=531)
        wineRFModel.fit(train_set_x, train_set_y)
        
        #MSE accumulation for data sets
        prediction = wineRFModel.predict(test_set_x)
        mseOos.append(mean_squared_error(test_set_y, prediction))
    print("MSE")
    print(mseOos[-1])
    
    #Draw a diagram of the number of trees in the ensemble against the training test error
    plt.plot(nTreeList, mseOos)
    plt.xlabel('Number of maxFeat in Ensemble')
    plt.ylabel('Mean Squared Error')
    plt.show()

def find_max_features(train_set_x,train_set_y,test_set_x,test_set_y):
    
    #Random forest training in the size range of the ensemble to check for changes in MSE
    mseOos = []
    nTreeList = range(1, 23, 1)
    for iTrees in nTreeList:
        wineRFModel = RandomForestRegressor(n_estimators=30,max_depth=17, max_features=iTrees,oob_score=False, random_state=531)
        wineRFModel.fit(train_set_x, train_set_y)
        
        #MSE accumulation for data sets
        prediction = wineRFModel.predict(test_set_x)
        mseOos.append(mean_squared_error(test_set_y, prediction))
    print("MSE")
    print(mseOos[-1])
    
    #Draw a diagram of the number of trees in the ensemble against the training test error
    plt.plot(nTreeList, mseOos)
    plt.xlabel('Number of max_features in Ensemble')
    plt.ylabel('Mean Squared Error')
    plt.show()

def MSE(fit,train_set_x,train_set_y,test_set_x,test_set_y,p):
    #Mean Squared Error
    print("MSE: %.2f" % np.sqrt(np.mean((p - test_set_y) ** 2)))
    # Explained variance score: 1 is perfect prediction
    print('Train Variance score: %.2f' % fit.score(train_set_x, train_set_y))
    print('Test Variance score: %.2f' % fit.score(test_set_x, test_set_y))

def make_csv(predict_y,test_set_x,test_set_y):
    ori = test_set_y.reset_index(drop=True)
    tmp = concat(test_set_x.reset_index(drop=True),ori)
    tmp = concat(tmp,pd.DataFrame({'predict_price':predict_y}))
    #tmp = concat(tmp,pd.DataFrame({'difference':ori-predict_y}))
    tmp = concat(tmp,pd.DataFrame({'error':(ori-predict_y)/ori*100}))
    tmp.predict_price = tmp.predict_price.astype(int)
    tmp.error = tmp.error.astype(int)
    tmp.to_csv("./csv/tested.csv", mode='w', index=False)

def plot_feature_importances(train_set_x,model):
    #print("feature_importances : \n{}".format(model.feature_importances_))
    n_features = train_set_x.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), train_set_x.columns)
    plt.ylim(-1, n_features)
    plt.show()

def map(test_set_x,test_set_y,predic):
    import folium
    LDN_COORDINATES = (39.9375346, 115.837023)
    myMap = folium.Map(location=LDN_COORDINATES, zoom_start=10)
    for i in range(1,500):
        lat = test_set_x['Lat'].iloc[i]
        lng = test_set_x['Lng'].iloc[i]
        p_price = str(predic[i]).split('.')[0]
        price = str(test_set_y.iloc[i])
        folium.Marker([lat, lng],popup='<i>original_price: ' + price + '\npredict_price: ' + p_price + '</i>').add_to(myMap)
    myMap.save('./map.html')

#find_estimators(train_set_x,train_set_y,test_set_x,test_set_y)
#find_max_depth(train_set_x,train_set_y,test_set_x,test_set_y)
#find_max_features(train_set_x,train_set_y,test_set_x,test_set_y)

print('\nRandomForestRegressor:')
forest_reg = RandomForestRegressor(n_estimators = 30, max_depth = 17, max_features=20,bootstrap=True)
forest_reg.fit(train_set_x, train_set_y)
predict_y = forest_reg.predict(test_set_x)

MSE(forest_reg,train_set_x,train_set_y,test_set_x,test_set_y,predict_y)
plot_feature_importances(test_set_x,forest_reg)

make_csv(predict_y,test_set_x,test_set_y)
map(test_set_x,test_set_y,predict_y)
########
