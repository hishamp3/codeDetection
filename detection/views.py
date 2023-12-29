from django.shortcuts import render
from . import models


def getlabel(code) -> str:
    l1 = models.LLM()
    token_code = l1.tokenize(code)
    prediction = l1.detection(token_code)
    return prediction


def detector(request):
    if request.method == 'POST':
        code = request.POST['code']
        prediction_value = getlabel(code)
        return render(request, 'index.html',
                      {'prediction_value': prediction_value, 'code': code})
    return render(request, 'index.html')
