# Load libraries
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib as mpl 

import pyspark as pys
from pyspark.sql import SQLContext

sc = pys.SparkContext('local', 'pyspark tutorial') 
sqlContext = SQLContext(sc)
sc.addPyFile('pyspark_csv.py')
import pyspark_csv as pycsv

plaintext_rdd = sc.textFile('loan_t1.csv')
dataframe = pycsv.csvToDataFrame(sqlContext, plaintext_rdd, sep=";")
print(dataframe.describe())




# train_filename = "loan_t1.csv"
# train_data = pd.read_csv(train_filename, sep=";", encoding="ISO-8859-1")
# X_train = train_data.loc[:,'loan_amnt':]
# y_train = train_data.loc[:,['loan_status']]
# y_train_np = np.array(y_train).reshape(len(y_train),)
# pd.set_option('display.width', 100)
# # # class distribution for the test dataset
# print(y_train.groupby('loan_status').size())
# pd.set_option('precision', 2)
# print(X_train.describe())
