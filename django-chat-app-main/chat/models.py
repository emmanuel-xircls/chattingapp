from django.db import models
from datetime import datetime

# Create your models here.
class Room(models.Model):
    name = models.CharField(max_length=1000)
    def __str__(self):
        return self.name
class Message(models.Model):
    value = models.CharField(max_length=1000000)
    date = models.DateTimeField(default=datetime.now, blank=True)
    user = models.CharField(max_length=1000000)
    room = models.ForeignKey(Room, on_delete=models.CASCADE)    

    def __str__(self):
        return self.user,self.value
    
class DashboardEntry(models.Model):
    room_name = models.CharField(max_length=1000)
    user_name = models.CharField(max_length=1000)
    created_at = models.DateTimeField(auto_now_add=True)
    has_unread_message = models.BooleanField(default=False)


    def __str__(self):
        return f"Room: {self.room_name}, User: {self.user_name}"