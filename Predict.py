#!/usr/bin/env python
# coding: utf-8

# In[29]:


"""
This Python code snippet serves two purposes:
1. illustrates how to use relative path
2. provides the template for code submission
ASSUMPTION: 
1. This Python code is present in the folder 'srika_DS_456AB'.
2. BMTC.parquet.gzip, Input.csv, and GroundTruth.csv are present in the folder 'data'
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from math import cos, asin, sqrt, pi
from sklearn import tree
import copy
import os
# import other packages here
from math import radians, cos, sin, asin, sqrt
pd.options.mode.chained_assignment = None  # default='warn'

"""
ILLUSTRATION: HOW TO USE RELATIVE PATH
Given the above mentioned assumptions, when you run the code, the following three commands will read the files 
containing data, input and, the ground truth.
"""
df = pd.read_parquet('./../data/BMTC.parquet.gzip', engine='pyarrow') # This command loads BMTC data into a dataframe. 
                                                                      # In case of error, install pyarrow using: 
                                                                      # pip install pyarrow
dfInput = pd.read_csv('./../data/Input.csv')
dfGroundTruth = pd.read_csv('./../data/GroundTruth.csv') 
# NOTE: The file GroundTruth.csv is for participants to assess the performance their own codes

"""
CODE SUBMISSION TEMPLATE
1. The submissions should have the function EstimatedTravelTime().
2. Function arguments:
    a. df: It is a pandas dataframe that contains the data from BMTC.parquet.gzip
    b. dfInput: It is a pandas dataframe that contains the input from Input.csv
3. Returns:
    a. dfOutput: It is a pandas dataframe that contains the output
"""
def EstimatedTravelTime(df, dfInput): # The output of this function will be evaluated
    # Function body - Begins
    # Make changes here.
    dfOutput = pd.DataFrame()
    if not os.path.isfile('./../data/extract_data.csv'):
        extract_data,extract_label = extract_data_from_parquet(df)
        extract_data.to_csv('./../data/extract_data.csv')
        extract_label.to_csv('./../data/extract_label.csv')
    else:
        extract_data = pd.read_csv('./../data/extract_data.csv')
        extract_label = pd.read_csv('./../data/extract_label.csv')
    kmeans,nc = cluster_model(extract_data,nc=300)
    if not os.path.isfile('./../data/preprocess_df.csv'):
        x_train = preprocess(extract_data,kmeans,nc)
        x_train.to_csv('./../data/preprocess_df.csv')
    else:
        x_train = pd.read_csv('./../data/preprocess_df.csv')
    x_test = preprocess(dfInput,kmeans,nc)
#     knn = neighbors.KNeighborsRegressor(n_neighbors=11)
#     knn = knn.fit(x_train.values,extract_label)
#     y_pred = knn.predict(x_test.values)
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(x_train, extract_label)
    y_pred = clf.predict(x_test)
    
    # Function body - Ends
    dfOutput = copy.deepcopy(dfInput)
    dfOutput['ETT'] = y_pred
    return dfOutput 
  
"""
Other function definitions here: BEGINS
"""
def extract_data_from_parquet(data):
    # for evry bus id in BusID column
    train_data = pd.DataFrame(columns=['BusID','Source_Lat','Source_Long','Dest_Lat','Dest_Long','TT','index_'])
    import warnings
    for i in data['BusID'].unique():
        # filter data for that particular bus id
        temp = data[data['BusID'] == i]
        for j in temp.index:
            # for every index in the filtered data
            # check if the time difference is less than 1 minute
            if j+1 in temp.index and temp.loc[j+1,'Speed']!=0 and (temp.loc[j+1,'Timestamp'] - temp.loc[j,'Timestamp']).seconds >=40 and (temp.loc[j+1,'Timestamp'] - temp.loc[j,'Timestamp']).seconds <= 3600:
                # print((temp.loc[j+1,'Timestamp'] - temp.loc[j,'Timestamp']).seconds,'index=',j,j+1,'BusID=',i)
                # if yes then append the data to train_data
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=FutureWarning)
                    train_data = train_data.append({'BusID':int(i),'Source_Lat':temp.loc[j,'Latitude'],'Source_Long':temp.loc[j,'Longitude'],'Dest_Lat':temp.loc[j+1,'Latitude'],'Dest_Long':temp.loc[j+1,'Longitude'],'TT':(temp.loc[j+1,'Timestamp'] - temp.loc[j,'Timestamp']).seconds/60,'index_':(j,j+1)},ignore_index=True)
    labels = train_data['TT']
    train_data = train_data.drop(['BusID','index_','TT'],axis=1)
    return train_data,labels
def cluster_model(ip,nc=300):
    coords = np.vstack((ip[['Source_Lat',  'Source_Long']].values,
                        ip[['Dest_Lat', 'Dest_Long']].values,
                        ip[['Source_Lat',  'Source_Long']].values,
                        ip[['Dest_Lat', 'Dest_Long']].values))
    kmeans = MiniBatchKMeans(n_clusters=nc, batch_size=100).fit(coords)
    return kmeans,nc
def haversine_distance(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a)) #2*R*asin...
def preprocess(df,kmeans,nc):
    ip = copy.deepcopy(df)
    ip['haversine_distance'] = ip.apply(lambda x: haversine_distance(x['Source_Lat'],  x['Source_Long'],x['Dest_Lat'], x['Dest_Long']), axis=1)
    for i in range(nc):
        ip[f'pickup_cluster_{i+1}'] = np.zeros(len(ip))
        ip[f'dropoff_cluster{i+1}'] = np.zeros(len(ip))
    for index in ip.index:
        x  = ip.iloc[index]
        p1 = kmeans.predict([[x['Source_Lat'],  x['Source_Long']]])
        x[f'pickup_cluster_{p1}'] = 1
        p2 = kmeans.predict([[x['Dest_Lat'], x['Dest_Long']]])
        x[f'dropoff_cluster_{p2}'] = 1
    return ip

"""
Other function definitions here: ENDS
"""

dfOutput = EstimatedTravelTime(df, dfInput)

