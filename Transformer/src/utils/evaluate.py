import pdb

def stack_to_string(stack):
	op = ""
	for i in stack:
		if op == "":
			op = op + i
		else:
			op = op + ' ' + i
	return op

def cal_score(outputs, target):
	w = 0
	lens = 0
	#pdb.set_trace()
	for i in range(len(outputs)):
		wrong = 0
		op = outputs[i]
		tgt = target[i].split()
		c = 0
		l = 0
		for j in range(len(tgt)):
			c+=1
			try:
				if tgt[j] != op[j]:
					wrong+=1
			except:
				wrong+=1
		while c < len(op):
			wrong+=1
			c+=1
			l+=1
		w += wrong
		lens += len(tgt)
		lens+=l
	return w/lens

def cal_score_AP(outputs, target):
    w = 0
    lens = 0
    #pdb.set_trace()
    for i in range(len(outputs)):
        wrong = 0
        op = outputs[i].split('_')
        #print(op)
        tgt = target[i].split('_')
        #print(tgt)
        c = 0
        for j in range(len(tgt)):
            c+=1
            try:
                if tgt[j] != op[j]:
                    wrong+=1
            except:
                wrong+=1
        l=0
        while c < len(op):
            wrong+=1
            c+=1
            l+=1
        w += wrong
        lens += len(tgt)
        lens += l
    return w/lens


