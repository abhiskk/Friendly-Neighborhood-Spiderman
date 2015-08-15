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
import heapq


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

    def getids(self):
        """Returns ids of datapoints present in features"""
        return list(self.features.keys())

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
            feature_vec = {}
            for word in self.bowcollection[instanceid]:
                id = word[0]
                val = float(word[1])
                val_tfidf = val * math.log(doc_count / float(count[id]), 2)
                if val_tfidf > self.tfidf_threshold:
                    feature_vec[id] = val_tfidf
            self.features[instanceid] = feature_vec

    def cosine_similarity(self, u, v):
        """calculates (u . v) / (mod(u) * mod(v))"""
        modu = math.sqrt(sum([u[x] ** 2 for x in u.keys()]))
        modv = math.sqrt(sum([v[x] ** 2 for x in v.keys()]))
        if modu == 0 or modv == 0:
            return 0
        dotproduct = 0.0
        for key in u.keys():
            if key in v:
                dotproduct += u[key] * v[key]
        return dotproduct / (modu * modv)

    def similarity(self, u, v):
        """Finds similarity for feature vectors 'u' and 'v'.
        Currently cosine similarity is used."""
        return self.cosine_similarity(u, v)

    def knn(self, instanceid, nearestk=5):
        """Finds k nearest neighbors for 'instanceid' datapoint"""
        assert instanceid in self.features
        featurevector = self.features[instanceid]
        neighbors = []
        for id in self.features.keys():
            if id == instanceid:
                continue
            val = self.similarity(featurevector, self.features[id])
            if len(neighbors) < nearestk:
                heapq.heappush(neighbors, (val, id))
            elif val > neighbors[0][0]:
                heapq.heappop(neighbors)
                heapq.heappush(neighbors, (val, id))
        L = len(neighbors)
        A = [heapq.heappop(neighbors) for x in range(L)]
        return [x[1] for x in reversed(A)]


def tests():
    nn = NearestNeighbor()
    nn.addinstance("test.txt")
    nn.addinstance(title="New York", instancedata="")
    assert abs(nn.similarity({0: 1.0, 2: 2.0}, {0: 1.0, 2: 2.0}) - 1) < 1e-6
    assert abs(nn.similarity({0: 1.0, 2: 2.0}, {
               0: 5.5}) - 5.5 / (5 ** 0.5 * 5.5)) < 1e-6
    return "Tests pass."


def main():
    logging.basicConfig(filename="nearestneighbor.log", level=logging.DEBUG)
    nn = NearestNeighbor()
    print("Adding instances")
    nn.addbulkinstances("Data/")
    print("Creating tfidf features")
    nn.create_tfidf_features()

if __name__ == "__main__":
    main()
