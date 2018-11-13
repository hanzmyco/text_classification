import sys

def check(label_file,result_file):
    labels=[]
    results=[]
    for line in open(label_file):
        line = line.strip()
        labels.append(line)
    for line in open(result_file):
        line = line.strip()
        results.append(line)
    l = len(labels)
    num_correct = 0
    for index in range(0,l):
        if labels[index] == results[index]:
            num_correct +=1
    print(num_correct)
    print(l)

check(sys.argv[1],sys.argv[2])
