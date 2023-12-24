from django.shortcuts import render
from . import models


def getlabel(code) -> str:
    l1 = models.LLM()
    return "vulnerable"


def detector(request):
    if request.method == 'POST':
        code = request.POST['code']
        prediction_value = getlabel(code)
        return render(request, 'main.html',
                      {'prediction_value': prediction_value, 'code': code})
    return render(request, 'main.html')
