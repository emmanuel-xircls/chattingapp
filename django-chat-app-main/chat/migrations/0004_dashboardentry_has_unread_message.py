# Generated by Django 4.2.1 on 2023-06-14 03:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0003_dashboardentry'),
    ]

    operations = [
        migrations.AddField(
            model_name='dashboardentry',
            name='has_unread_message',
            field=models.BooleanField(default=False),
        ),
    ]
