from sklearn.model_selection import train_test_split
# read_in data_total as a file, generate 10 sets of different training and testing sets

def k_fold_validation(data_total,label_total,train_files_out,test_files_out,train_labels_out,test_labels_out):
    data_all=[]
    label_all=[]

    for line in open(data_total, encoding='utf-8'):
        data_all.append(line.strip())
    for line in open(label_total):
        label_all.append(line.strip())

    # randomly split data and label into 2 sets, one for train, one for test

    for index in range(0,10):
        X_train,X_test,y_train,y_test = train_test_split(data_all,label_all,test_size=0.1)

        with open(train_files_out+'.'+str(index), 'w', encoding='utf-8') as f:
            for ite in X_train:
                f.write(ite+'\n')

        with open(test_files_out+'.'+str(index), 'w', encoding='utf-8') as f:
            for ite in X_test:
                f.write(ite+'\n')

        with open(train_labels_out+'.'+str(index), 'w', encoding='utf-8') as f:
            for ite in y_train:
                f.write(ite+'\n')

        with open(test_labels_out+'.'+str(index), 'w', encoding='utf-8') as f:
            for ite in y_test:
                f.write(ite+'\n')









