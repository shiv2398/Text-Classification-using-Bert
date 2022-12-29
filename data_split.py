from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical
def data_split(data):
    """
    function : data_split(data)---------------------------->

    This function help to split the data into train and test with 80/20 proportion
  
    Argument : data 
    

    * y_raw ->  is the raw sentiment to validate the predicigton in the end 
    * y_train,y_test -> changed into the categorical
    
    return x_train,x_test,y_train,y_test,y_raw
    """
    x_train,x_test,y_train,y_test=train_test_split(data['Phrase'],data['Sentiment'].values,test_size=0.2)
    y_raw=y_test
    y_train=to_categorical(y_train)
    y_test=to_categorical(y_test)
    return x_train,x_test,y_train,y_test,y_raw

