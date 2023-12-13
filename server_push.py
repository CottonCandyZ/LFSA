
import urllib.parse
import urllib.request
def sc_send(text, desp=''):
    key = "" # ios BARK push key
    postdata = urllib.parse.urlencode({'title': text, 'body': desp}).encode('utf-8')
    url = f'https://api.day.app/{key}'
    req = urllib.request.Request(url, data=postdata, method='POST')
    with urllib.request.urlopen(req) as response:
        result = response.read().decode('utf-8')
    return result