
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.sparse import csr_matrix, hstack
from datetime import date
mingw_path = 'C:\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

import xgboost as xgb

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils


# In[ ]:

def rstr(df): return df.dtypes, df.head(3) ,df.apply(lambda x: [x.unique()]), df.apply(lambda x: [len(x.unique())]),df.shape


# In[ ]:

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


# In[ ]:

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0


# In[ ]:

seed = 1851
np.random.seed(seed)


# In[ ]:

datadir = ''
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),
                      index_col='device_id')
gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),
                     index_col = 'device_id')


# In[ ]:

gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])


# In[ ]:

gatrain.head()


# In[ ]:

phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
phone.head()


# In[ ]:

phone.shape


# In[ ]:

phone.duplicated().sum()


# In[ ]:

phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
phone.shape


# In[ ]:

phone.head()


# In[ ]:

events = pd.read_csv(os.path.join(datadir,'events.csv'),
                     parse_dates=['timestamp'])
events.head()


# In[ ]:

appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'),
                        usecols=['event_id','app_id','is_active'],
                        dtype={'is_active':bool})
appevents.head()


# In[ ]:

applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))
applabels.head()


# In[ ]:

labelcategories = pd.read_csv(os.path.join(datadir,'label_categories.csv'))
labelcategories.head()


# In[ ]:

print('Number of unique categores: {}, Number of unique categores in apps {}, Number of categories that match {}'.format(labelcategories.shape[0],len(applabels.label_id.unique()),labelcategories.label_id.isin(applabels.label_id).sum()))


# # Device Brand

# In[ ]:

brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']
Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.brand)))
Xte_brand = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.brand)))
print('Brand features: train shape {}, test shape {}'.format(Xtr_brand.shape, Xte_brand.shape))


# # Device Model

# In[ ]:

m = phone.phone_brand.str.cat(phone.device_model)
m.head()


# In[ ]:

modelencoder = LabelEncoder().fit(m)
phone['model'] = modelencoder.transform(m)
gatrain['model'] = phone['model']
gatest['model'] = phone['model']
Xtr_model = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.model)))
Xte_model = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.model)))
print('Model features: train shape {}, test shape {}'.format(Xtr_model.shape, Xte_model.shape))


# In[ ]:

appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)
napps = len(appencoder.classes_)
appevents.head()


# In[ ]:

events.head()


# In[ ]:

latLong = pd.read_csv("latLong.csv")
latLong.head()


# In[ ]:

events = events.merge(latLong[['longitude','latitude','regionLabel']],how='left',left_on=['longitude','latitude'],right_on=['longitude','latitude'])
events.head()


# In[ ]:

events = events.drop(['longitude','latitude'],axis=1)
events.head()


# In[ ]:

events.dtypes


# In[ ]:

events['day_of_week']=events.timestamp.dt.dayofweek
events['time_of_day']=events.timestamp.dt.hour
events['period']=pd.cut(events.time_of_day,bins=[0,5,12,17,21,24],right=False,labels=[0,1,2,3,4])
events['period'].replace(4,0,inplace=True)
# 0:Night,1:Morning,2:Afternoon,3:Evening
events =events.drop(['time_of_day','timestamp'],axis=1)
events.head()


# In[ ]:

appevents.head()


# In[ ]:

appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)
napps = len(appencoder.classes_)
appevents.head()


# In[ ]:

deviceapps=appevents.merge(events,how='left',left_on=['event_id'],right_on=['event_id'])
deviceapps.head()


# # Apps and Activity

# In[ ]:

appactivity = deviceapps.groupby(['device_id','app'])['is_active'].agg(['size',np.sum,np.mean]).reset_index()
appactivity=(appactivity.merge(gatrain[['trainrow']],how='left',left_on='device_id',right_index=True)
.merge(gatest[['testrow']],how='left',left_on='device_id',right_index=True))
appactivity.head()


# In[ ]:

d = appactivity.dropna(subset=['trainrow'])
Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)), 
                      shape=(gatrain.shape[0],napps))
d = appactivity.dropna(subset=['testrow'])
Xte_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)), 
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_app.shape, Xte_app.shape))


# In[ ]:

d = appactivity.dropna(subset=['trainrow'])
Xtr_app_size = csr_matrix((d['size'], (d.trainrow, d.app)), 
                      shape=(gatrain.shape[0],napps))
d = appactivity.dropna(subset=['testrow'])
Xte_app_size = csr_matrix((d['size'], (d.testrow, d.app)), 
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_app_size.shape, Xte_app_size.shape))


# In[ ]:

d = appactivity.dropna(subset=['trainrow'])
Xtr_app_act = csr_matrix((np.log1p(d['sum']), (d.trainrow, d.app)), 
                      shape=(gatrain.shape[0],napps))
d = appactivity.dropna(subset=['testrow'])
Xte_app_act = csr_matrix((np.log1p(d['sum']), (d.testrow, d.app)), 
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_app_act.shape, Xte_app_act.shape))


# In[ ]:

d = appactivity.dropna(subset=['trainrow'])
Xtr_app_act_m = csr_matrix((d['mean'], (d.trainrow, d.app)), 
                      shape=(gatrain.shape[0],napps))
d = appactivity.dropna(subset=['testrow'])
Xte_app_act_m = csr_matrix((d['mean'], (d.testrow, d.app)), 
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_app_act_m.shape, Xte_app_act_m.shape))


# # Location and Activity

# In[ ]:

locactivity = deviceapps.groupby(['device_id','regionLabel'])['is_active'].agg(['size',np.sum,np.mean]).reset_index()
locactivity=(locactivity.merge(gatrain[['trainrow']],how='left',left_on='device_id',right_index=True)
.merge(gatest[['testrow']],how='left',left_on='device_id',right_index=True))
locactivity.head()


# In[ ]:

locEncoder = LabelEncoder().fit(locactivity['regionLabel'])
locactivity['loc'] = locEncoder.transform(locactivity['regionLabel'])
nlocs = len(locEncoder.classes_)
locactivity.head()


# In[ ]:

d = locactivity.dropna(subset=['trainrow'])
Xtr_loc = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d['loc'])), 
                      shape=(gatrain.shape[0],nlocs))
d = locactivity.dropna(subset=['testrow'])
Xte_loc = csr_matrix((np.ones(d.shape[0]), (d.testrow, d['loc'])), 
                      shape=(gatest.shape[0],nlocs))
print('Apps data: train shape {}, test shape {}'.format(Xtr_loc.shape, Xte_loc.shape))


# In[ ]:

d = locactivity.dropna(subset=['trainrow'])
Xtr_loc_size = csr_matrix((d['size'], (d.trainrow, d['loc'])), 
                      shape=(gatrain.shape[0],nlocs))
d = locactivity.dropna(subset=['testrow'])
Xte_loc_size = csr_matrix((d['size'], (d.testrow, d['loc'])), 
                      shape=(gatest.shape[0],nlocs))
print('Apps data: train shape {}, test shape {}'.format(Xtr_loc_size.shape, Xte_loc_size.shape))


# In[ ]:

d = locactivity.dropna(subset=['trainrow'])
Xtr_loc_act = csr_matrix((np.log1p(d['sum']), (d.trainrow, d['loc'])), 
                      shape=(gatrain.shape[0],nlocs))
d = locactivity.dropna(subset=['testrow'])
Xte_loc_act = csr_matrix((np.log1p(d['sum']), (d.testrow, d['loc'])), 
                      shape=(gatest.shape[0],nlocs))
print('Apps data: train shape {}, test shape {}'.format(Xtr_loc_act.shape, Xte_loc_act.shape))


# In[ ]:

d = locactivity.dropna(subset=['trainrow'])
Xtr_loc_act_m = csr_matrix((d['mean'], (d.trainrow, d['loc'])), 
                      shape=(gatrain.shape[0],nlocs))
d = locactivity.dropna(subset=['testrow'])
Xte_loc_act_m = csr_matrix((d['mean'], (d.testrow, d['loc'])), 
                      shape=(gatest.shape[0],nlocs))
print('Apps data: train shape {}, test shape {}'.format(Xtr_loc_act_m.shape, Xte_loc_act_m.shape))


# # Week day and Activity

# In[ ]:

weekactivity = deviceapps.groupby(['device_id','day_of_week'])['is_active'].agg(['size',np.sum,np.mean]).reset_index()
weekactivity=(weekactivity.merge(gatrain[['trainrow']],how='left',left_on='device_id',right_index=True)
.merge(gatest[['testrow']],how='left',left_on='device_id',right_index=True))
weekactivity.head()


# In[ ]:

d = weekactivity.dropna(subset=['trainrow'])
Xtr_week = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d['day_of_week'])), 
                      shape=(gatrain.shape[0],7))
d = weekactivity.dropna(subset=['testrow'])
Xte_week = csr_matrix((np.ones(d.shape[0]), (d.testrow, d['day_of_week'])), 
                      shape=(gatest.shape[0],7))
print('Apps data: train shape {}, test shape {}'.format(Xtr_week.shape, Xte_week.shape))


# In[ ]:

d = weekactivity.dropna(subset=['trainrow'])
Xtr_week_size = csr_matrix((d['size'], (d.trainrow, d['day_of_week'])), 
                      shape=(gatrain.shape[0],7))
d = weekactivity.dropna(subset=['testrow'])
Xte_week_size = csr_matrix((d['size'], (d.testrow, d['day_of_week'])), 
                      shape=(gatest.shape[0],7))
print('Apps data: train shape {}, test shape {}'.format(Xtr_week_size.shape, Xte_week_size.shape))


# In[ ]:

d = weekactivity.dropna(subset=['trainrow'])
Xtr_week_act = csr_matrix((np.log1p(d['sum']), (d.trainrow, d['day_of_week'])), 
                      shape=(gatrain.shape[0],7))
d = weekactivity.dropna(subset=['testrow'])
Xte_week_act = csr_matrix((np.log1p(d['sum']), (d.testrow, d['day_of_week'])), 
                      shape=(gatest.shape[0],7))
print('Apps data: train shape {}, test shape {}'.format(Xtr_week_act.shape, Xte_week_act.shape))


# In[ ]:

d = weekactivity.dropna(subset=['trainrow'])
Xtr_week_act_m = csr_matrix((d['mean'], (d.trainrow, d['day_of_week'])), 
                      shape=(gatrain.shape[0],7))
d = weekactivity.dropna(subset=['testrow'])
Xte_week_act_m = csr_matrix((d['mean'], (d.testrow, d['day_of_week'])), 
                      shape=(gatest.shape[0],7))
print('Apps data: train shape {}, test shape {}'.format(Xtr_week_act_m.shape, Xte_week_act_m.shape))


# # Day time & Activity

# In[ ]:

periodactivity = deviceapps.groupby(['device_id','period'])['is_active'].agg(['size',np.sum,np.mean]).reset_index()
periodactivity=(periodactivity.merge(gatrain[['trainrow']],how='left',left_on='device_id',right_index=True)
.merge(gatest[['testrow']],how='left',left_on='device_id',right_index=True))
periodactivity.head()


# In[ ]:

d = periodactivity.dropna(subset=['trainrow'])
Xtr_period = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d['period'])), 
                      shape=(gatrain.shape[0],4))
d = periodactivity.dropna(subset=['testrow'])
Xte_period = csr_matrix((np.ones(d.shape[0]), (d.testrow, d['period'])), 
                      shape=(gatest.shape[0],4))
print('Apps data: train shape {}, test shape {}'.format(Xtr_period.shape, Xte_period.shape))


# In[ ]:

d = periodactivity.dropna(subset=['trainrow'])
Xtr_period_size = csr_matrix((d['size'], (d.trainrow, d['period'])), 
                      shape=(gatrain.shape[0],4))
d = periodactivity.dropna(subset=['testrow'])
Xte_period_size = csr_matrix((d['size'], (d.testrow, d['period'])), 
                      shape=(gatest.shape[0],4))
print('Apps data: train shape {}, test shape {}'.format(Xtr_period_size.shape, Xte_period_size.shape))


# In[ ]:

d = periodactivity.dropna(subset=['trainrow'])
Xtr_period_act = csr_matrix((np.log1p(d['sum']), (d.trainrow, d['period'])), 
                      shape=(gatrain.shape[0],4))
d = periodactivity.dropna(subset=['testrow'])
Xte_period_act = csr_matrix((np.log1p(d['sum']), (d.testrow, d['period'])), 
                      shape=(gatest.shape[0],4))
print('Apps data: train shape {}, test shape {}'.format(Xtr_period_act.shape, Xte_period_act.shape))


# In[ ]:

d = periodactivity.dropna(subset=['trainrow'])
Xtr_period_act_m = csr_matrix((d['mean'], (d.trainrow, d['period'])), 
                      shape=(gatrain.shape[0],4))
d = periodactivity.dropna(subset=['testrow'])
Xte_period_act_m = csr_matrix((d['mean'], (d.testrow, d['period'])), 
                      shape=(gatest.shape[0],4))
print('Apps data: train shape {}, test shape {}'.format(Xtr_period_act_m.shape, Xte_period_act_m.shape))


# In[ ]:

del appactivity
del locactivity
del weekactivity
del periodactivity
del events


# # App labels features

# In[ ]:

applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]
applabels['app'] = appencoder.transform(applabels.app_id)
labelencoder = LabelEncoder().fit(applabels.label_id)
applabels['label'] = labelencoder.transform(applabels.label_id)
nlabels = len(labelencoder.classes_)


# In[ ]:

# del appevents
gatrain.head()


# In[ ]:

#devicelabels = (deviceapps[['device_id','app']]
#                .merge(applabels[['app','label']])
#                .groupby(['device_id','label'])['app'].agg(['size'])
#                .merge(gatrain[['trainrow']], how='left',  left_index=True, right_index=True)
#                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
#               .reset_index())
devicelabels = pd.read_csv(os.path.join(datadir,'devicelabels.csv'))
devicelabels.head()


# In[ ]:

d = devicelabels.dropna(subset=['trainrow'])
Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)), 
                      shape=(gatrain.shape[0],nlabels))
d = devicelabels.dropna(subset=['testrow'])
Xte_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)), 
                      shape=(gatest.shape[0],nlabels))
print('Labels data: train shape {}, test shape {}'.format(Xtr_label.shape, Xte_label.shape))


# # Join all features

# In[ ]:

Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label, Xtr_loc, Xtr_period, Xtr_loc_act_m, Xtr_period_act_m, Xtr_loc_act, Xtr_period_act), format='csr')
Xtest =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label, Xte_loc, Xte_period, Xte_loc_act_m, Xte_period_act_m, Xte_loc_act, Xte_period_act), format='csr')
print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))


# In[ ]:

Xtrain_log = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label, Xtr_loc, Xtr_period), format='csr')
Xtest_log =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label, Xte_loc, Xte_period), format='csr')
print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))


# # Cross-validation

# In[ ]:

def baseline_model():
    # create model
    model = Sequential()
    #model.add(Dense(10, input_dim=Xtrain.shape[1], init='normal', activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(50, input_dim=Xtrain.shape[1], init='normal', activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(12, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model


# In[ ]:

targetencoder = LabelEncoder().fit(gatrain.group)
y = targetencoder.transform(gatrain.group)
nclasses = len(targetencoder.classes_)


# In[ ]:

params = {}
params['booster'] = 'gblinear'
params['objective'] = "multi:softprob"
params['eval_metric'] = 'mlogloss'
params['eta'] = 0.01
params['num_class'] = 12
params['lambda'] = 5
params['alpha'] = 3


# In[ ]:

parameters = {'C':0.02,'l1_ratio':1}


# In[ ]:

model=baseline_model()


# In[ ]:




# In[ ]:

def score(parameters, random_state = 0,mix={'logit':0.475,'xgb':0.475,'sgd':0.05}):
    kf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=random_state)
    pred = np.zeros((y.shape[0],nclasses))
    pred_logistic = np.zeros((y.shape[0],nclasses))
    pred_xgb = np.zeros((y.shape[0],nclasses))
    pred_sgd = np.zeros((y.shape[0],nclasses))
    logit_val = False
    xgb_val = False
    sgd_val = False
    if mix['logit']>0:
        logit_val = True
    if mix['xgb']>0:
        xgb_val = True
    if mix['sgd']>0:
        sgd_val = True
    for itrain, itest in kf:
        Xtr, Xte = Xtrain[itrain, :], Xtrain[itest, :]
        #Xtr_log, Xte_log = Xtrain_log[itrain, :], Xtrain_log[itest, :]
        ytr, yte = y[itrain], y[itest]
        if logit_val:
            print("Starting Logistic Regression")
            # Logistic Regression
            clf1 = LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs')
            clf1.fit(Xtr, ytr)
            pred_logistic[itest,:] = clf1.predict_proba(Xte)
        if xgb_val:
            print("Starting XGBoost")
            # XGBoost
            d_train = xgb.DMatrix(Xtr, label=ytr)
            d_valid = xgb.DMatrix(Xte, label=yte)
            watchlist = [(d_train, 'train'), (d_valid, 'eval')]
            clf2 = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=10,verbose_eval=False)
            pred_xgb[itest,:] = clf2.predict(d_valid)
        if sgd_val:
            # SGD Classifier
            print("SGD Classifier")
            clf3 = SGDClassifier(loss='log',penalty='elasticnet',l1_ratio=parameters['l1_ratio'])
            clf3.fit(Xtr, ytr)
            pred_sgd[itest,:] = clf3.predict_proba(Xte)
        # Downsize to one fold only for kernels
        # Combine predictions
        pred[itest,:] = (mix['logit']*pred_logistic[itest,:])+(mix['xgb']*pred_xgb[itest,:])+(mix['sgd']*pred_sgd[itest,:])
        #return log_loss(yte, pred[itest, :])
        print("Logistic {:.5f}, XGB {:.5f}, SGD {:.5f}, Average {:.5f}".format(log_loss(yte, pred_logistic[itest,:]),log_loss(yte, pred_xgb[itest,:]),log_loss(yte, pred_sgd[itest,:]),log_loss(yte, pred[itest,:])), end=' ')
        print('')
    print('')
    return log_loss(y, pred)


# In[ ]:

Cs = np.logspace(-3,0,4)
res = []
for C in Cs:
    res.append(score(LogisticRegression(C = C)))
plt.semilogx(Cs, res,'-o');


# In[ ]:

score(LogisticRegression(C=0.02))


# In[ ]:

score(parameters)


# In[ ]:

score(parameters,mix=0.75)


# In[ ]:

score(parameters)


# In[ ]:

score(parameters={'C':0.02,'l1_ratio':0.5},mix={'logit':0.475,'xgb':0.475,'sgd':0.05})


# In[ ]:

regularization = np.arange(0.5,0.7,0.05)
res = []
for l1_ratio in regularization:
    parameters['l1_ratio'] = l1_ratio
    res.append(score(parameters,mix={'logit':0,'xgb':0,'sgd':1}))
for l1_ratio,scores in zip(regularization,res):
    print("Regularization {:.2f}, Score {:.5f}".format(l1_ratio,scores))


# In[ ]:

score(parameters,mix={'logit':0.5,'xgb':0.5,'sgd':0})


# In[ ]:

clf_logit = LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs')
clf_logit.fit(Xtrain,y)
pred1 = clf_logit.predict_proba(Xtest)


# In[ ]:

d_train = xgb.DMatrix(Xtrain, label=y)
d_valid = xgb.DMatrix(Xtest)
watchlist = [(d_train, 'train')]
clf_xgb = xgb.train(params, d_train, 300, watchlist, early_stopping_rounds=10,verbose_eval=True)
pred2 = clf_xgb.predict(d_valid)


# In[ ]:

pred = pred1*0.5 + pred2*0.5
pred = pd.DataFrame(pred, index = gatest.index, columns=targetencoder.classes_)
pred.head()


# In[ ]:

pred.to_csv('logit_xgb_subm.csv',index=True)

