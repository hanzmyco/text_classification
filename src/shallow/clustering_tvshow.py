import config
import read_json
import Vectorizer
import KMeans
import Algorithm
import BaseModel
import tv_show

def main():

    data_in=[]
    feed_id=[]
    print('start reading data')
    path = 'E:\\QQ_Browser_data\\ruyizhuan.csv'
    path2 = 'E:\\QQ_Browser_data\\yanxigonglue.csv'
    tv_show.process_data(path,feed_id,data_in)
    tv_show.process_data(path2,feed_id,data_in)




    if config.mode =='Training':
        if config.model_name == 'Counter':
            model = Vectorizer.CounterVector(config.model_name)
        elif config.model_name == 'TfIdf':
            model = Vectorizer.TfIdfVector(config.model_name)
            print('finish initilizing model')
        elif config.model_name =='FeatureHasher':
            model = Vectorizer.FeatureHasherVector(config.model_name,config.n_features)

        model.feature_transform(data_in)
        print(len(model.vectorizer.vocabulary_))


        if config.algo_name =='KMeans':
            algo_instance = KMeans.KMeansClustering(config.algo_name)
            print('start training model')
            algo_instance.fit(model.feature)
            algo_instance.serilize_model()
            print('finish serilizing model')
            algo_instance.output_cluster_info(data_in,model,feed_id)




    else:
        print('loading vectorizer')
        model=BaseModel.BaseModel(config.model_name)
        model.de_serilize_model()
        print('finish loading vector')


        if config.algo_name =='KMeans':
            algo_instance = Algorithm.Base_Algorithm(config.algo_name)
            algo_instance.de_serilize_model()
            print('finish desirialization')
            features=model.transform(data_in)

            labels=algo_instance.predict(features)
            print(labels)
            #algo_instance.get_centroids()
            #algo_instance.output_cluster_info(data_in, model, feed_id)
            print('finish all')


if __name__ == '__main__':
    main()
