# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import stats
from sklearn import metrics
import random
from tensorflow import keras
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth=True
session=InteractiveSession(config=config)
import seaborn as sns
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense,Dropout,BatchNormalization,LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import tensorflow as tf


# In[2]:


n_dim=5   #number of dimensions
###########load data
data = pd.read_csv('training_dataset.csv')
data = data.fillna(0)
data = np.array(data)

data[:,0]*=10 ### C element * 10

all_input = data[:,0:n_dim] #element content
all_SDI = data[:,n_dim] #property 1

#round
round_num  = 1
round_num2 = round_num-1
round_name = 'Round'+str(round_num)
round_name2 = 'Round'+str(round_num2)
#######################
n_model=5#number of models
n_model2=20#repeat 20 times, means 5*20 = 100 different models
#######################################################
model_folder = "Results"
# Check if the directory exists
if not os.path.exists(model_folder):
    # If it doesn't exist, create it
    os.makedirs(model_folder)
if not os.path.exists(model_folder+'/'+round_name):
    # If it doesn't exist, create it
    os.makedirs(model_folder+'/'+round_name)
if not os.path.exists(model_folder+'/'+round_name2):
    # If it doesn't exist, create it
    os.makedirs(model_folder+'/'+round_name2)


# In[3]:


#training the performance prediction model
def train_model(X, y,i,model_name):
    ind=index_random[round(i*len(index_random)/n_model):round((1+i)*len(index_random)/n_model)]####1/5 data as test set
    ind2=np.setdiff1d(index_random, ind)
    X_train, X_test, y_train, y_test = X[ind2],X[ind], y[ind2],y[ind]
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, strides=1, padding='same', activation='elu', input_shape=(n_dim, 1)))
    model.add(LayerNormalization())
    model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', activation='elu'))
    model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', activation='elu'))
    model.add(Conv1D(8, kernel_size=3, strides=1, padding='same', activation='elu'))
    model.add(Conv1D(4, kernel_size=3, strides=1, padding='same', activation='elu'))
    model.add(Flatten())
    model.add(Dense(128,activation='elu'))
    model.add(Dense(64,activation='elu'))
    model.add(Dense(32,activation='elu'))
    model.add(Dense(1,activation='linear'))
    model.summary()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=["mean_squared_error"])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=500)
    mc = ModelCheckpoint(model_folder+'/'+round_name2+f"/model_{model_name}.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    history=model.fit(X_train, y_train, validation_data=(X_test, y_test),  batch_size=128, epochs=5000, callbacks=[es,mc])
    model=keras.models.load_model(model_folder+'/'+round_name2+f"/model_{model_name}.h5")
    R2,MAE=mae_r2(model,X_test,y_test)
    return model,X_test,y_test,R2,MAE

def custom_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return y_true * mse

def mae_r2(model,X_test,y_test):
    y_pred = model.predict(X_test.reshape(len(X_test),n_dim,1))
    R=stats.pearsonr(y_pred.reshape(-1), y_test.reshape(-1))[0]
    R2=R**2
    R2=np.asarray(R2).round(6)
    MAE= metrics.mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))
    return R2,MAE

def model_performance(model,X_test,y_test):
    perform_list=pd.read_csv(model_folder+'/'+round_name2+'/'+'model_performance_SDI.csv')
    y_pred = model.predict(X_test.reshape(len(X_test),n_dim, 1))
    R=stats.pearsonr(y_pred.reshape(-1), y_test.reshape(-1))[0]
    R2=R**2
    R2=np.asarray(R2).round(6)
    MAE  = metrics.mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))
    MAPE = np.mean(np.abs((y_test.reshape(-1)-y_pred.reshape(-1)) / y_test.reshape(-1))) * 100
    MSE = np.mean((y_test.reshape(-1)-y_pred.reshape(-1))**2)
    RMSE = np.sqrt(np.mean((y_test.reshape(-1)-y_pred.reshape(-1))**2))
    ###plot R2 and MAE of test data
    plt.figure()
    sns.set()
    sns.regplot(x=y_pred, y=y_test, color='k') 
    plt.title(('R2:',R2,'MAE:',MAE,'MAPE:',MAPE,'MSE:',MSE,'RMSE:',RMSE))
    y_test = pd.DataFrame(y_test)
    y_test.columns= ['ground truth']
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns= ['pred']
    Metrics = pd.DataFrame([R2,MAE,MAPE,MSE,RMSE])
    Metrics.columns= ['Metrics']
    perform_list2=pd.concat((perform_list,y_test,y_pred,Metrics),axis=1)
    perform_list2.drop([perform_list2.columns[0]],axis=1, inplace=True)
    perform_list2.to_csv(model_folder+'/'+round_name2+'/'+'model_performance_SDI.csv')
    return R2,MAE,MAPE,MSE,RMSE


# In[4]:


###emsemble all models to predict
def emsemble_pred(SDI,n_model):#prediction of property 1
    pred_all=np.zeros((len(SDI),0))
    for n in range(n_model2):
        for m in range(n_model):
            i = n * n_model + m
            print("SDI shape(ensembel_pred1):", SDI.shape)
            temp=models[model_list1[i]].predict(SDI.reshape(len(SDI),n_dim,1))
            pred_all=np.concatenate((pred_all,temp.reshape(-1,1)),axis=1)
            
    mean = np.mean(pred_all,axis=1).reshape(-1,1)
    std = np.std(pred_all,axis=1).reshape(-1,1)
    
    return np.concatenate((mean, std), axis=1)

def normalize_data(data):
    mean_vals = np.mean(data, axis=0)
    std_vals = np.std(data, axis=0)
    std_vals[std_vals == 0] = 1
    normalized_data = (data - mean_vals) / std_vals
    return normalized_data

# In[ ]:


####train n models
pd.DataFrame(np.empty(0)).to_csv(model_folder+'/'+round_name2+'/model_performance_SDI.csv')
metrics_list=[]

for n in range(n_model2):
    #########slice the data to five parts
    index_random=np.arange(len(all_input))
    random.shuffle(index_random)
    index_random1=index_random[:]
    
    for m in range(n_model):
        i = n * n_model + m
        trytime=0
        model,X_test,y_test,R21,MAE = train_model(all_input, all_SDI ,m,f'SDI{i}')
        R20=R21
        while trytime < 5:
            trytime+=1
            model1,X_test1,y_test1,R21,MAE1 = train_model(all_input, all_SDI ,m,f'SDI{i+100}')#train the model of ahc
            if R21>R20:
                R20=R21
                model1.save(model_folder+'/'+round_name2+f"/model_SDI{i}.h5")
        model1=keras.models.load_model(model_folder+'/'+round_name2+f"/model_SDI{i}.h5")
        R21,MAE1,MAPE,MSE,RMSE=model_performance(model1,X_test1,y_test1)#show and save the performance of the model
        metrics_list.append([R21,MAE1,MAPE,MSE,RMSE])


# In[ ]:

####metrics output
metrics_list = np.array(metrics_list)
metrics_list2 = pd.DataFrame(metrics_list)
metrics_list2.columns= ['R2','MAE','MAPE','MSE','RMSE']
metrics_list2.to_csv(model_folder+'/'+round_name2+'/'+'metrics.csv',index = False)


# In[ ]:
    
####load all models and store in a dict "models"
path = os.getcwd()
name2=path+'/'+model_folder+'/'+round_name2+'/'
models=dict()
model_list1=[]
for i in range(n_model * n_model2):
    modelname = f'model_SDI{i}'
    model_list1.append(modelname)
    models[modelname]= keras.models.load_model(name2+modelname+'.h5')


# In[ ]:


####initial predict
initial_pred = emsemble_pred(all_input,n_model)
data[:,0]/=10 ### C element * 10

all_initial = np.concatenate((data,initial_pred),axis=1)
all_initial2 = pd.DataFrame(all_initial)
all_initial2.columns= ['C','Mn','Al','Si','Fe','real SDI','pred SDI mean','std']
all_initial2.to_csv(model_folder+'/all_initial_pred.csv',index = False)






