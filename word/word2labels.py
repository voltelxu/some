import re

def word2lables(filename, outfile):
	file = open(filename, 'r')
	out = open(outfile, 'w+')
	data = ""
	for line in file:
		line = line.strip('\n')
		words = line.split(" ")
		for word in words:
			if len(word) < 3:
				continue
			if len(word) == 3:
				out.write(word + " S\n")
				data = data + word + " S\n"
				continue
			if len(word) > 3:
				for i in range(0, len(word), 3):
					if i == 0:
						out.write(word[i:3] + " B\n")
						data = data + word[i:3] + " B\n"
					elif i == len(word) - 3:
						out.write(word[i:i+3] + " E\n")
						data = data + word[i:i+3] + " E\n"
					else:
						out.write(word[i:i+3] + " M\n")
						data = data +word[i:i+3] + " M\n"
	#out.write(data)
	out.close()
	file.close()

word2lables('data', 'lables')
