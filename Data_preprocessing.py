import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.utils import shuffle

def data_prep(file_name):
    """
    function : data_prep()---------------------------------->

    This function take the path of the file and make the data bias by maintaining the equal(7354) prportion of the every class and return it --
    
    Argument : file_path
    return Non_bias_data """
    train_data=pd.read_table(file_name)
    e_class_label_count=min(train_data.Sentiment.value_counts())
    Non_bias_train_data=pd.DataFrame()
    for i in (train_data['Sentiment'].unique()):
        temp_labels=train_data[train_data['Sentiment']==i]
        Non_bias_train_data=pd.concat([Non_bias_train_data,temp_labels[:e_class_label_count]])
    data=shuffle(Non_bias_train_data)
    return data




