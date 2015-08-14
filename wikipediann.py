# -----------------------------------------------------------------------------
# Author: Abhishek Kadian (abhishekkadiyan@gmail.com)
# -----------------------------------------------------------------------------

"""Wikipedia Nearest Neighbor:
Finds the nearest neighbor of a Wikipedia article.
"""

import os
import time
import logging
import wikipedia
import re


class NearestNeighbor:

    def __init__(self, boost_title=1):
        # TODO: add the docstring
        self.data = []
        self.dic = {}
        self.bowcollection = {}
        self.boost_title = boost_title

    def bagofwords(self, s, title=None):
        """Creates bag of words representation for s.
        Each word is represented as [word,wordcount] where wordcount is the
        number of times word occurs in the document
        """
        bow = {}
        if not title is None:
            words = list(filter(len, re.split("\W", title)))
            for word in words:
                if not word in bow:
                    bow[word] = self.boost_title
                else:
                    bow[word] += self.boost_title
        for line in s:
            line = line.lower()
            words = list(filter(len, re.split("\W", line)))
            for word in words:
                if not word in bow:
                    bow[word] = 1
                else:
                    bow[word] += 1
        return bow

    def updatedictionary(self, bow):
        """Adds any new words present in bow to self.dic"""
        for w in bow:
            if not w in self.dic:
                self.dic[w] = len(self.dic)

    def addinstance(self, instanceid, filepath=None, title=None, instancedata=None):
        """Reads data from file or takes in a string. Converts the instance
        into bag of words and stores in data. Pass the title in-case you
        want the model to give extra weight to title
        """
        assert filepath != None or instancedata != None
        if filepath != None:
            with open(filepath, "r", encoding="utf-8") as f:
                instancedata = f.readlines()
        bow = self.bagofwords(instancedata, title)
        self.updatedictionary(bow)
        self.bowcollection[instanceid] = bow

    def addbulkinstances(self, datafolder):
        """Calls 'addinstance' on all the files present in datafolder"""
        for filepath in os.listdir(datafolder):
            self.addinstance(instanceid=filepath,
                             filepath=datafolder + filepath)

    def calculatetfidf(self):
        """Calculated tf-idf for all the bag of words stored in self.bowcollection"""
        raise NotImplementedError


def download(titlesfile, datafolder="data/"):
    """Reads list of titles from 'filepath', downloads wikipedia articles for
    titles and stores them as separate files in datafolder.
    """
    if not os.path.exists(datafolder):
        os.makedirs(datafolder)
    with open(titlesfile, "r", encoding="utf-8") as f:
        for line in f:
            title = line.strip()
            logging.debug("Downloading \'{0}\'".format(title.encode("utf-8")))
            page = wikipedia.page(title)
            with open(datafolder + title + ".txt", "wb") as g:
                g.write(bytes(page.title + "\n", "utf-8"))
                g.write(bytes(page.content, "utf-8"))


def tests():
    nn = NearestNeighbor()
    nn.addinstance("test.txt")
    nn.addinstance(title="New York", instancedata="")
    return "Tests pass."


def main():
    logging.basicConfig(filename="wikipediann.log", level=logging.DEBUG)
    nn = NearestNeighbor()
    nn.addbulkinstances("TestData/")
    nn.calculatetfidf()

if __name__ == "__main__":
    main()
