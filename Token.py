import re

class Token(object): 

    def __init__(self, text, tag=""):
        self.text = text
        self.features = []
        self.tag = tag

    def get_text(self):
        return self.text

    def get_tag(self):
        return self.tag

    def generateFeatures(self):
        self.features.append("W=" + self.getText()) # feature word itself
        self.features.append("SUFF3=" + self.getText()[-3:]) # feature suffix - last 3 chars
        self.features.append("SUFF2=" + self.getText()[-2:]) # feature suffix - last 2 chars
        self.features.append("PREF3=" + self.getText()[0:3]) # feature prefix - first 3 chars
        self.features.append("PREF2=" + self.getText()[0:2]) # feature prefix - first 2 chars
        if re.search(r"^[A-Z]+$", self.getText()): # the word written in caps
            self.features.append("ALLCAP=yes")
        else:
            self.features.append("ALLCAP=no")
        if re.search(r"^[A-Z].*", self.getText()): # the word begins with a cap
            self.features.append("BEGINCAP=yes")
        else:
            self.features.append("BEGINCAP=no")
        if re.search(r"^\d+[/.,\d]*$", self.getText()): # the word is a number (CD)
            self.features.append("DIGIT=yes")
        else:
            self.features.append("DIGIT=no")
        if re.search(r"-", self.getText()): # the word contains hyphen
            self.features.append("HYPHEN=yes")
        else:
            self.features.append("HYPHEN=no")


    def getFeatures(self):
        return self.features
