import json
import requests

def api_get_request(url):
    # In this exercise, you want to call the last.fm API to get a list of the
    # top artists in Spain.
    url = 'http://ws.audioscrobbler.com/2.0/?method=geo.gettopartists&country=spain&api_key=b7f14a12b6ab167b145008cc16f0f1f2&format=json'
    
    data = requests.get(url).text
    data=json.loads(data)
    
    
   
    # Once you've done this, return the name of the number 1 top artist in Spain.
    
    return data['topartists']['artist'][0]['name']
# return the top artist in Spain

