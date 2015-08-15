# -----------------------------------------------------------------------------
# Author: Abhishek Kadian (abhishekkadiyan@gmail.com)
# -----------------------------------------------------------------------------

"""nearestneighbor:
Finds the nearest neighbor for a text article using tf-idf model
and cosine similarity
"""

import os
import math
import time
import logging
import re
import heapq
import wikipedia


class NeighborFinder:

    def __init__(self, boost_title=1, tfidf_threshold=0):
        """Increase the value of boost_title to improve the weightage
        given to title of article. Features with weight below tfidf_threshold
        are neglected.
        """
        self.data = []
        self.idword = {}
        self.bowcollection = {}
        self.features = {}
        self.tfidf_count = {}
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
        W = []
        if type(s) == str:
            x = s.lower()
            W = list(filter(len, re.split("\W+", x)))
        else:
            for line in s:
                line = line.lower()
                words = list(filter(len, re.split("\W+", line)))
                for w in words:
                    W.append(w)
        for word in W:
            if not word in self.idword:
                ind = len(self.idword)
                self.idword[word] = len(self.idword)
            else:
                ind = self.idword[word]
            if not ind in dic:
                dic[ind] = 1
            else:
                dic[ind] += 1
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
        doc_count = len(self.bowcollection)
        for instanceid in self.bowcollection.keys():
            for word in self.bowcollection[instanceid]:
                id = word[0]
                if not id in self.tfidf_count:
                    self.tfidf_count[id] = 1
                else:
                    self.tfidf_count[id] += 1
        for instanceid in self.bowcollection.keys():
            featurevector = {}
            for word in self.bowcollection[instanceid]:
                id = word[0]
                val = float(word[1])
                val_tfidf = val * \
                    math.log(doc_count / float(self.tfidf_count[id]), 2)
                if val_tfidf > self.tfidf_threshold:
                    featurevector[id] = val_tfidf
            self.features[instanceid] = featurevector

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

    def knn(self, title=None, instanceid=None, nearestk=3):
        """Finds k nearest neighbors for 'instanceid' datapoint.
        If 'title' is passed then the wikipedia article corresponding
        to 'title' is downloaded and nearest neighbors for 'title' article
        are predicted.
        """
        assert title != None or instanceid != None
        if title == None:
            assert instanceid in self.features
            featurevector = self.features[instanceid]
        elif title + ".txt" in self.features.keys():
            instanceid = title + ".txt"
            featurevector = self.features[title + ".txt"]
        else:
            instanceid = title + ".txt"
            page = wikipedia.page(title)
            content = page.content.lower()
            words = list(filter(len, re.split("\W+", content)))
            wordcounts = {}
            for word in words:
                if word in self.idword:
                    id = self.idword[word]
                    if id in wordcounts:
                        wordcounts[id] += 1
                    else:
                        wordcounts[id] = 1
            featurevector = {}
            for id in wordcounts.keys():
                val = wordcounts[id]
                val_tfidf = val * \
                    math.log(len(self.bowcollection) /
                             float(self.tfidf_count[id]), 2)
                if val_tfidf > self.tfidf_threshold:
                    featurevector[id] = val_tfidf

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

    def updatedataset(self, instances):
        """'instances' is a list of tuples of instance-id instance-text.
        Adds instances to the datapool and recomputes the tf-idf feature
        vectors
        """
        for instance in instances:
            self.addinstance(instanceid=instance[0], instancedata=instance[1])
        self.features = {}
        self.tfidf_count = {}
        self.create_tfidf_features()


def main():
    logging.basicConfig(filename="nearestneighbor.log", level=logging.DEBUG)
    nn = NeighborFinder()
    print("Adding instances")
    nn.addbulkinstances("Data/")
    print("Creating tfidf features")
    nn.create_tfidf_features()
    print("Updating dataset")
    T = ["Ryan Giggs", "Sharukh Khan"]
    R = [[t, wikipedia.page(t).content] for t in T]
    nn.updatedataset(R)
    print("Predicting")
    with open("output.txt", "w") as f:
        # Note that articles in L are already present in the training dataset
        L = ["Lionel Messi.txt", "Breaking Bad.txt",
             "Google.txt", "John Cena.txt", "Eminem.txt", "Donald Trump.txt",
             "Deadpool.txt", "Salman Khan.txt"]
        for x in L:
            f.write(x + ": " + str(nn.knn(instanceid=x)) + "\n")
        f.write("-" * 50 + "\n")
        # Note that articles in A are downloaded on the fly hence the
        # prediction is slower
        A = ["Wayne Rooney", "Football", "Winston Churchill"]
        for x in A:
            f.write(x + ": " + str(nn.knn(title=x)) + "\n")


if __name__ == "__main__":
    main()
