import numpy as np
import pickle
import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer,TFBertModel
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
import transformers
model_load = tf.keras.models.load_model('optimized_model_2.2.h5', custom_objects={"TFBertModel": transformers.TFBertModel})

encoded_dict={
     'Negative':0,
    'Somewhat Negative':1,
    'Neutral':2,
    'Somewhat Positive':3,
    'Positive':4
}
def get_max_ac(d):
  return max(d, key = d.get)
def predict(model,tokenizer,input_text):
    x_val = tokenizer(
        text=input_text,
        add_special_tokens=True,
        max_length=70,
        truncation=True,
        padding='max_length', 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True) 
    output_dict={}
    validation = model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
    for key , value in zip(encoded_dict.keys(),validation[0]):
        output_dict[key]=value
    return output_dict
st.title('Sentiment Analysis on Movie Reviews! ! ')
st.info("This application aims to classify the movies reviews. Write a simple sentence (not more than 70 words).")
input_text = st.text_area('Enter Text Below (maximum 70 words):', height=100)
submit=st.button('Predict')
if submit:
    st.subheader("Probabilities :")
    with st.spinner(text="This may take a moment..."):
        output=predict(model_load,tokenizer,input_text) 
    result=get_max_ac(output)
    for key,probab in output.items():
        st.write(key,probab)
    
    st.subheader("Result :")
    st.write('Movie Review is : ',result)

