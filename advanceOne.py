# Load libraries
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib as mpl 
from pyspark import SparkContext
from pyspark.sql import SQLContext
sc = SparkContext('local', 'pyspark') 
sqlContext = SQLContext(sc)
plaintext_rdd=sc.textFile("loan_t1.csv") \
	.map(lambda line: line.split(";")) \
	.filter(lambda line: len(line)>1) \
	.map(lambda line: (line[0],line[1])) \
	.collect()
df = sqlContext.read.load('loan_t1.csv', format='com.databricks.spark.csv', header='true', inferSchema='true')
print(rdd_temp_K)  	
#plaintext_rdd = sc.textFile('loan_t1.csv')
#lines_rdd = sc.textFile('loan_t1.csv')
#rdd_temp_K =lines_rdd.map(lambda x: x + 273).take(3)
# dataframe = pycsv.csvToDataFrame(sqlContext, plaintext_rdd, sep=";")
# data_sort_np=np.array(dataframe)	
# for index,item_data in enumerate(data_sort_np):
# 	print(item_data)
#print(dataframe.describe())
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
