with open("frame_types.txt", 'rt', encoding='UTF8') as f:
	body = f.readlines()
	num = 0
	for line in body:
		if line.rstrip()[-1] == "I":
			num += 1
	print(num)
	f.close()