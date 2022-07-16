from django import forms
from .models import Music

class MusicForm(forms.ModelForm):
    url = forms.CharField(max_length=1000)
    class Meta:
        model = Music
        fields = ['url']