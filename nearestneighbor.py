# -----------------------------------------------------------------------------
# Author: Abhishek Kadian (abhishekkadiyan@gmail.com)
# -----------------------------------------------------------------------------

"""NearestNeighbor:
Finds the nearest neighbor for a text article using tf-idf model
and cosine similarity
"""

import os
import time
import logging
import re
import crawler


class NearestNeighbor:
    # TODO: add docstring

    def __init__(self, boost_title=1):
        # TODO: add docstring
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
            words = list(filter(len, re.split("\W+", title)))
            for word in words:
                if not word in bow:
                    bow[word] = self.boost_title
                else:
                    bow[word] += self.boost_title
        for line in s:
            line = line.lower()
            words = list(filter(len, re.split("\W+", line)))
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


def tests():
    nn = NearestNeighbor()
    nn.addinstance("test.txt")
    nn.addinstance(title="New York", instancedata="")
    return "Tests pass."


def main():
    logging.basicConfig(filename="nearestneighbor.log", level=logging.DEBUG)
    nn = NearestNeighbor()
    nn.addbulkinstances("TestData/")
    nn.calculatetfidf()

if __name__ == "__main__":
    main()
