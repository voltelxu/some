

class Lable():
    feature = {}
    stopword = []
    def __init__(self):
        lines = open('/root/clim/scrapy/tutorial/tutorial/spiders/cat/allfeature', 'r').read().split('\n')
        #print len(lines)
        for i in range(len(lines) - 1):
            w,d = lines[i].split('#')
            self.feature[w] = eval(d)
        lines = open('/root/clim/scrapy/tutorial/tutorial/spiders/cat/stopword', 'r').read().split('\n')
        for i in range(len(lines) - 1):
            self.stopword.append(lines[i])
        print("feature and stop word load successful!")
