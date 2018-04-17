out=open('../data/labels_all.txt','a+')
for i in range(0,19469):
    if i % 2 ==0:
        out.write('1\n')
    else:
        out.write('0\n')
