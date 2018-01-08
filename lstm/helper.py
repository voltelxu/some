import numpy as np
import random

class data():
    index = 0
    data = []
    def getdic(self, filename):
        dic = open(filename,'r').read().split('\n')
        self.voc = dict()
        for i in range(len(dic) -1):
            word = dic[i].split(' ')[0]
            index = dic[i].split(' ')[1]
            self.voc[word] = int(index)

    def getdata(self, filename):
        self.getdic('dic')
        self.index = 0
        files = open(filename, 'r')
        self.max_len = 0
        for l in files:
            data = []
            lable,raw = l.strip('\n').split('xkx')
            raw = raw.split(' ')
            if len(raw) > self.max_len:
                self.max_len = len(raw)
            dataid = [self.voc.get(word,0) for word in raw]
            data.append(dataid)
            data.append(int(lable))
            self.data.append(data)
        random.shuffle(self.data)
    def getbatch(self, batchsize):
        batch = self.data[self.index:self.index + batchsize]
        self.index = self.index + batchsize
        if self.index > len(self.data):
            self.index = 0
        batchdata = [b[0] for b in batch]
        seq_len = []
        for d in batchdata:
            seq_len.append(len(d))
            d += [0 for _ in range(self.max_len - len(d))]
        lable = [b[1] for b in batch]
        return batchdata, lable, seq_len
