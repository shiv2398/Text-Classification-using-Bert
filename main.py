from Data_preprocessing import data_prep
from data_split import data_split
from model_data_preprocessing import train_test_prepare
from model import bert_model
from train_model import train_model
from test_model import test_model
from transformers import AutoTokenizer,TFBertModel
import tensorflow as tf
import transformers
import pickle
from max_key import get_max_ac

file_name=('train.tsv.zip')
def main():
    #tokenizer 
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    #bold the fonts in the print function
    RED    = "\033[31m"
    BOLD   = "\033[1m"
    print(BOLD+'Bert Sentiment Analysis ! !')
    func_details_=input("Do You want to know the function details(yes/no):")
    #fucntion details conditon
    if func_details_=='yes':
        print('\nFunction Details')
        print(data_prep.__doc__)
        print(data_split.__doc__)
        print(train_test_prepare.__doc__)
        print(bert_model.__doc__)
        print(train_model.__doc__)
        print(test_model.__doc__)
    x=input("\nYou want to train the model(yes/no):")
    if x=='yes':
        #data preprocessing function
        data=data_prep(file_name)
        #dataspliting function
        x_train,x_test,y_train,y_test,y_raw=data_split(data)
        
        print(f'x_train_shape:{x_train.shape},\nx_test_shape:{x_test.shape},\ny_train_shape:{y_train.shape},\ny_test_shape:{y_test.shape}')
        #bert model data preparation
        x_train,x_test,input_ids,attention_mask=train_test_prepare(x_train,x_test,tokenizer)
        #length of the sentence
        max_length=70
        #Defining the bert model
        b_model=bert_model(max_length,tokenizer)

        inp=input('\nYou want to save the model(yes/no):')
        if inp=='yes':
            save_=True
        else:
            save_=False
        # training the bert model
        model=train_model(b_model,x_train,x_test,y_train,y_test,"Optimized_model",epochs=10,save_model=save_,plot_accuracy=True,plot_loss=True)
        test_model(model,y_raw,x_test)
    else:
        print('\nLoading the availaible model :===================>\n')
        #loading the saved model
        model= tf.keras.models.load_model('optimized_model_2.2.h5', custom_objects={"TFBertModel": transformers.TFBertModel})
    # sentiment class dict
    encoded_dict={
        'Negative':0,
        'Somewhat Negative':1,
        'Neutral':2,
        'Somewhat Positive':3,
        'Positive':4}
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    #test the input
    texts = input(str('\nEnter the text You want to analyse : '))
    # txt post preprocessing
    x_val = tokenizer(
        text=texts,
        add_special_tokens=True,
        max_length=70,
        truncation=True,
        padding='max_length', 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True) 
    output_dict={}
    print('\nPredicting :--->\n')
    validation = model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
    for key , value in zip(encoded_dict.keys(),validation[0]):
        #saving into dictionary
        output_dict[key]=value
        print(f'\n{key} -> Probability : {value}')
    #max_probability
    result=get_max_ac(output_dict)
    RED    = "\033[31m"
    print(f'\n So the review is : \033[1m{RED+result}\033[0m')

if __name__ == "__main__":
    main()

    
