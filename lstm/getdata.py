import re
import os
import math

def eachdir(path):
    pathdir = os.listdir(path)
    dic = list()
    for p in pathdir:
        f = os.path.join('%s/%s' %(path, p))
        dic.append(f)
    return dic

def getfiles(filepath):
    path = os.listdir(filepath)
    filenames = list()
    for p in path:
        filename = os.path.join('%s/%s' %(filepath, p))
        filenames.append(filename)
    return filenames

def getdata(filename):
#    print filename
    data = open(filename, 'r').read().decode('euc-kr')
    titles = re.findall('<title>(.*?)</title>', data, re.S)
    subtitles = re.findall('<subtitle>(.*?)</subtitle>', data, re.S)
    bodys = re.findall('<body>(.*?)</body>', data, re.S)
    title = ""
    subtitle = ""
    body = ""
    for t in titles:
        #t = unicode(t, 'utf-8')
        t = t.replace('\n', ' ')
        t = re.sub(u'[^\uAC00-\uD7A3 ]', '', t)
        t = re.sub('[ ]+', ' ', t)
        title = title + t
    for t in subtitles:
        #t = unicode(t, 'utf-8')
        t = t.replace('\n', ' ')
        t = re.sub(u'[^\uAC00-\uD7A3 ]', '', t)
        t = re.sub('[ ]+', ' ', t)
        subtitle = subtitle + t
    for t in bodys:
        #t = unicode(t, 'utf-8')
        t = t.replace('\n', ' ')
        t = re.sub(u'[^\uAC00-\uD7A3 ]', '', t)
        t = re.sub('[ ]+', ' ', t)
        body = body + t
    return title.strip(' '), subtitle.strip(' '), body.strip(' ')



def predata(i, lable, filename):
    title, subtitle, body = getdata(filename)
    train = open('train','a')
    test = open('test', 'a')
    text = title.strip(' ')
    text = text.split(' ')
    title = ""
    for t in text:
        for j in range(len(t)):
            title = title + t[j:j+1] + " "
    title = title.strip(' ')
    if i <= 5001:
        train.write(str(lable) + "xkx" + title.encode('utf-8') + '\n')
    else:
        test.write(str(lable) + "xkx" + title.encode('utf-8') + '\n')
    train.close()
    test.close()

def main(data):
    child = eachdir(data)
    lables = {'15':0, '16':1, '21':2, '22':3, '23':4, '24':5, '25':6, '3B':7, '4':8, '52':9, 'G1':10, 'G2':11}
    for c in child:
        j = 0
        lable = lables[c[8:]]
        filenames = getfiles(c)
        for filename in filenames:
            predata(j, lable, filename)
            j = j + 1

main('../data')
