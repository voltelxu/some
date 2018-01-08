
def getdic():
    f = open('train', 'r').read().split('\n')
    voc = dict()
    voc['unk'] = 0
    for i in range(len(f) - 1):
        l = f[i]
        line = l.split('xkx')[1]
        line = line.split(' ')
        for word in line:
             pos = voc.get(word, -1)
             if pos == -1:
                 voc[word] = len(voc)
    dic = open('dic', 'w')
    for w in voc:
        dic.write(w + " " +str(voc[w]) + '\n')
    dic.close()
getdic()
