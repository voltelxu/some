import re
def  readfile(filename):
	file = open(filename, 'r')
	data = ''
	for line in file:
		data = data + line + " "
	file.close()
	return data

data = readfile('data')

def word2item(data):
	item = ""
	words = data.split(" ")
	for w in words:
		lst = re.split( '(\d+|\+|-|\*|/)', w)
		for content in lst:
			if(str(content).isdigit()):
				item = item + content + " "
			else:
				for i in range(0, len(content), 3):
					item = item + content[i: i + 3] + " "
	return item

item = word2item(data)
out = open('item', 'w')
out.write(item)
out.close()