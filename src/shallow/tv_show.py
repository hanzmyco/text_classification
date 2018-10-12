def process_data(input_path,feed_in,data_in):
    for line in open(input_path,'r',encoding='utf-8'):
        [vid,words]=line.strip().split(',')
        feed_in.append(vid)
        data_in.append(words)
        if words =='':
            break


def main():
    path = 'E:\\QQ_Browser_data\\ruyizhuan.csv'
    path2 = 'E:\\QQ_Browser_data\\yanxigonglue.csv'
    process_data(path2)


if __name__ == '__main__':
    main()