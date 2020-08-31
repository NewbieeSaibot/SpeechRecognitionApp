import requests

FILE_PATH = "./dataset/up/0a7c2a8d_nohash_0.wav"
URL = "http://18.191.135.104/predict"

if __name__ == '__main__':
    # open audio file
    file = open(FILE_PATH, 'rb')

    # create info for request
    values = {"file": (FILE_PATH, file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()

    # display predicted keyword
    print("Predicted Keyword: {}".format(data["keyword"]))
