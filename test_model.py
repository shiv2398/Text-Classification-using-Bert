import numpy as np
from sklearn.metrics import accuracy_score
def test_model(model,y_test,x_test):
    """function : test_model () ----------------------------------->
    This function helps in test the trained model and print the classification report and Accuracy 
    argument : model,y_test,x_test
    print classification report and Accuracy
    """
    predicted_raw = model.predict({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']})
    y_predicted = np.argmax(predicted_raw, axis = 1)
    from sklearn.metrics import classification_report
    print('Classification Matrix :')
    print(classification_report(y_test, y_predicted))
    print(f'\n Classification Accuracy')
    print(accuracy_score(y_test, y_predicted, normalize=False))
