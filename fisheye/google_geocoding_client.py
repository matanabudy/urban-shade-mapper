import requests

old = "https://maps.googleapis.com/maps/api/geocode/json?address=fDizengoff 23 Tel Aviv&key=AIzaSyBV_ngMP3bXIclCnK-nXMEq5imDQKVLrf0"


class GoogleGeoCodingClient():

    def __init__(self, token):
        self.endpoint = "https://maps.googleapis.com/maps/api/geocode/json?"
        self.token = token

    def get_address_coordinates(self, address):
        return requests.get(self.endpoint + "address='" + address + "'&key=" + self.token)
