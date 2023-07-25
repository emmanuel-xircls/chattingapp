from django.urls import path
from . import views

urlpatterns = [
    path('',views.bot,name="bot"),
    path('livechat/', views.checkview, name='home'),
    path('panel/', views.panel, name='panel'),
    path('<str:room>/', views.room, name='room'),
    path('send', views.send, name='send'),
    path('getMessages/<str:room>/', views.getMessages, name='getMessages'),
]