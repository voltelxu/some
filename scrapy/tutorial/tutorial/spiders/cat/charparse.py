from HTMLParser import HTMLParser

class Word_Parse():
    def parse(self, text):
        return HTMLParser().unescape(text)


