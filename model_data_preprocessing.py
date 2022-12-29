import pickle
def train_test_prepare(x_train,x_test,tokenizer):
    """ function : train_test_prepare()------------------------------------>
    Argument : x_train,x_test,tokenizer
   
    
    This function help to prepare the input data for the modek using the bert tokenizer

    * input_ids -> token indices for the input of the model
    
    * attention mask -> for the positional indices so that model do not consider them

    return : x_train,x_test,input_ids,attention_mask"""
    
    x_train = tokenizer(
        text=x_train.tolist(),
        add_special_tokens=True,
        max_length=70,
        truncation=True,
        padding=True, 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True)
    x_test = tokenizer(
        text=x_test.tolist(),
        add_special_tokens=True,
        max_length=70,
        truncation=True,
        padding=True, 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True)
    input_ids = x_train['input_ids']
    attention_mask = x_train['attention_mask']
   

    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return x_train,x_test,input_ids,attention_mask