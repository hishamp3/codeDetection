from django.urls import path
from . import views

urlpatterns = [
    path('', views.detector, name="detection"),
]
