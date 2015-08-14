# -----------------------------------------------------------------------------
# Author: Abhishek Kadian (abhishekkadiyan@gmail.com)
# -----------------------------------------------------------------------------

"""NearestNeighbor:
Finds the nearest neighbor for a text article using tf-idf model
and cosine similarity
"""

import os
import math
import time
import logging
import re
import crawler


class NearestNeighbor:
    # TODO: add docstring

    def __init__(self, boost_title=1, tfidf_threshold=0):
        # TODO: add docstring
        self.data = []
        self.idword = {}
        self.bowcollection = {}
        self.features = {}
        self.boost_title = boost_title
        self.tfidf_threshold = tfidf_threshold

    def bagofwords(self, s, title=None):
        """Creates bag of words representation for s.
        Each word is represented as [word,wordcount] where wordcount is the
        number of times word occurs in the document
        """
        dic = {}
        if not title is None:
            words = list(filter(len, re.split("\W+", title)))
            for word in words:
                if not word in self.idword:
                    ind = len(self.idword)
                    self.idword[word] = len(self.idword)
                else:
                    ind = self.idword[word]
                if not ind in dic:
                    dic[ind] = self.boost_title
                else:
                    dic[ind] += self.boost_title
        for line in s:
            line = line.lower()
            words = list(filter(len, re.split("\W+", line)))
            for word in words:
                if not word in self.idword:
                    ind = len(self.idword)
                    self.idword[word] = len(self.idword)
                else:
                    ind = self.idword[word]
                if not ind in dic:
                    dic[ind] = self.boost_title
                else:
                    dic[ind] += self.boost_title
        return [[w, dic[w]] for w in dic.keys()]

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
        self.bowcollection[instanceid] = bow

    def addbulkinstances(self, datafolder):
        """Calls 'addinstance' on all the files present in datafolder"""
        for filepath in os.listdir(datafolder):
            self.addinstance(instanceid=filepath,
                             filepath=datafolder + filepath)

    def create_tfidf_features(self):
        """Calculate tf-idf for all the bag of words stored in self.bowcollection"""
        count = {}
        doc_count = len(self.bowcollection)
        for instanceid in self.bowcollection.keys():
            for word in self.bowcollection[instanceid]:
                id = word[0]
                if not id in count:
                    count[id] = 1
                else:
                    count[id] += 1
        for instanceid in self.bowcollection.keys():
            feature_vec = []
            for word in self.bowcollection[instanceid]:
                id = word[0]
                val = float(word[1])
                val_tfidf = val * math.log(doc_count / float(count[id]), 2)
                if val_tfidf > self.tfidf_threshold:
                    feature_vec.append((id, val_tfidf))
            self.features[instanceid] = feature_vec



def tests():
    nn = NearestNeighbor()
    nn.addinstance("test.txt")
    nn.addinstance(title="New York", instancedata="")
    return "Tests pass."


def main():
    logging.basicConfig(filename="nearestneighbor.log", level=logging.DEBUG)
    nn = NearestNeighbor()
    nn.addbulkinstances("TestData/")
    nn.create_tfidf_features()

if __name__ == "__main__":
    main()
