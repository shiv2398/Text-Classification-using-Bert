from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
def train_model(model,x_train,x_test,y_train,y_test,model_name,epochs=10,save_model=True,plot_accuracy=True,plot_loss=True):
    """
    function : train_model () ---------------------------------------->

    This function is used to train the bert model

    Argumrent : model,x_train,x_test,y_train,y_test,model_name,epochs=10,save_model=True,plot_accuracy=True,plot_loss=True
   
    
    * model -> defined model
    * model_name -> for saving the model
    * epochs -> number of epochs
    * save_model -> for saving the model(True/False)
    * plot_accuracy -> to plot the accuracy(True/False)
    * plot_loss -> to plot the loss(True/False)
    
     return :  model and plot loss and accuracy
    """
    optimizer = Adam(
            learning_rate=5e-05, # this learning rate is for bert model , taken from huggingface website 
            epsilon=1e-08,
            decay=0.01,
            clipnorm=1.0)
        # Set loss and metrics
    loss =CategoricalCrossentropy(from_logits=True)
    metric = CategoricalAccuracy('balanced_accuracy'),
    print('GPU availabel:>',len(tf.config.list_physical_devices('GPU')))
    with tf.device('/device:GPU:0'):
        # Compile the model
        model.compile(
            optimizer = optimizer,
            loss = loss, 
            metrics = metric)
        train_history = model.fit(
            x ={'input_ids':x_train['input_ids'],'attention_mask':x_train['attention_mask']} ,
            y = y_train,
        validation_data = (
        {'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']}, y_test
        ),
      epochs=epochs,
      batch_size=36)
    if save_model:
        model.save(model_name,include_optimizer=False)
    if plot_accuracy:
        plt.figure(figsize=(10,10))
        plt.plot(train_history.history['balanced_accuracy'])
        plt.plot(train_history.history['val_balanced_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower right')
        plt.savefig('Model Accuracy')
        plt.show()
    if plot_accuracy:
        plt.figure(figsize=(10,10))
        plt.plot(train_history.history['loss'])
        plt.plot(train_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig('Model Loss')
        plt.show()
    return model 