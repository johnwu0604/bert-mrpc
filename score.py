import tensorflow as tf
from transformers import *
import json
from azureml.core import Model

def init():
    global tokenizer, model
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model_dir = Model.get_model_path('bert-mrpc')
    model = TFBertForSequenceClassification.from_pretrained(model_dir)
    
def run(raw_data):
    sentence_0 = json.loads(raw_data)['sentence_0']
    sentence_1 = json.loads(raw_data)['sentence_1']
    inputs = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors='tf')
    pred = model.predict(inputs)[0].argmax().item()
    result = ''
    if pred:
        result = " '{}' is a paraphrase of '{}' ".format(sentence_0, sentence_1)
    else:
        result = " '{}' is not a paraphrase of '{}' ".format(sentence_0, sentence_1)
    return { 'result': result }