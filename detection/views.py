from django.shortcuts import render
import torch
import os
# from transformers import AutoTokenizer, AutoModel, AdamW, AutoModelForSequenceClassification
from . import models

'''use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(os.getcwd() + "\\CodeBERT")
model_name = 'roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)'''


def getlabel(code) -> str:
    l1 = models.LLM()
    prediction = l1.detection(l1.tokenize(code))
    '''code_token = tokenizer(code, padding='max_length', max_length=512, truncation=True, return_tensors="pt").to(device)
    output = model(code_token['input_ids'], code_token['attention_mask'])
    logit = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    prediction = "vulnerability" if logit[0] == 0 else prediction = "Not vulnerability"'''
    return prediction


def detector(request):
    if request.method == 'POST':
        code = request.POST['code']
        #prediction_value = getlabel(code)
        return render(request, 'main.html',
                      {'prediction_value': "Vulnerable", 'code': code})
    return render(request, 'main.html')
