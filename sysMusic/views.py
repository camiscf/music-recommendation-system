from unittest import result
from django.shortcuts import render
# from .features import distance_onehot, getInputDF
# from .models import distance_onehot
from .features import getId, getInputDF, distance_onehot, main
from .forms import MusicForm

# Create your views here.
def index(request):
    if request.method == 'POST':
        form = MusicForm(request.POST)
        if form.is_valid():
            url = str(form.cleaned_data['url'])
            href_distance, music_distance, href_cosseno, music_cosseno = main(url)
            
            dados ={
                'form': form,
                'href_distance': href_distance,
                'music_distance': music_distance,
                'href_cosseno': href_cosseno,
                'music_cosseno': music_cosseno,
            }
        
            return render(request, 'home.html', dados)
        else:
            print('nao é valido')
            # return render(request, 'sysMusic/index.html', {'form': form})
    else:
        form = MusicForm()
        dados ={
            'form': form,
        }
    return render(request, 'home.html', {'form': form})
        # return render(request, 'sysMusic/index.html', {'form': form})


    # resultado = distance_onehot(getInputDF('https://open.spotify.com/track/1mQOFRDoqI34LdN79PE0vH?si=95b9eae1f9cd44e1'))
    # resultado = getId('https://open.spotify.com/track/1mQOFRDoqI34LdN79PE0vH?si=95b9eae1f9cd44e1')
    # resultado = getInputDF('https://open.spotify.com/track/1mQOFRDoqI34LdN79PE0vH?si=95b9eae1f9cd44e1') 

def about(request):
   return render(request, 'about.html')