import numpy as np
import pandas as pd
from recommenders.ISeqRecommender import ISeqRecommender
from util.data_utils import dataset_to_gru4rec_format
from util.rnn.gru4rec import GRU4Rec
from util.rnn.hgru4rec import HGRU4Rec
import csv

class RNNRecommender(ISeqRecommender):
    """
    A **simplified** interface to Recurrent Neural Network models for Session-based recommendation.
    Based on the following two papers:

    * Recurrent Neural Networks with Top-k Gains for Session-based Recommendations, Hidasi and Karatzoglou, CIKM 2018
    * Personalizing Session-based Recommendation with Hierarchical Recurrent Neural Networks, Quadrana et al, Recsys 2017

    """

    def __init__(self,
                 session_layers,
                 user_layers=None,
                 batch_size=32,
                 learning_rate=0.1,
                 momentum=0.0,
                 dropout=None,
                 epochs=10,
                 personalized=False):
        """
        :param session_layers: number of units per layer used at session level.
            It has to be a list of integers for multi-layer networks, or a integer value for single-layer networks.
        :param user_layers: number of units per layer used at user level. Required only by personalized models.
            It has to be a list of integers for multi-layer networks, or a integer value for single-layer networks.
        :param batch_size: the mini-batch size used in training
        :param learning_rate: the learning rate used in training (Adagrad optimized)
        :param momentum: the momentum coefficient used in training
        :param dropout: dropout coefficients.
            If personalized=False, it's a float value for the hidden-layer(s) dropout.
            If personalized=True, it's a 3-tuple with the values for the dropout of (user hidden, session hidden, user-to-session hidden) layers.
        :param epochs: number of training epochs
        :param personalized: whether to train a personalized model using the HRNN model.
            It will require user ids at prediction time.
        """
        super(RNNRecommender).__init__()
        if isinstance(session_layers, int):
            session_layers = [session_layers]
        if isinstance(user_layers, int):
            user_layers = [user_layers]
        self.session_layers = session_layers
        self.user_layers = user_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        if dropout is None:
            if not personalized:
                dropout = 0.0
            else:
                dropout = (0.0, 0.0, 0.0)
        self.dropout = dropout
        self.epochs = epochs
        self.personalized = personalized
        self.pseudo_session_id = 0

    def __str__(self):
        return 'RNNRecommender(' \
               'session_layers={session_layers}, ' \
               'user_layers={user_layers}, ' \
               'batch_size={batch_size}, ' \
               'learning_rate={learning_rate}, ' \
               'momentum={momentum}, ' \
               'dropout={dropout}, ' \
               'epochs={epochs}, ' \
               'personalized={personalized}, ' \
               ')'.format(**selfi.__dict__)
    
    
    
    def bert2dict(self,bert_filename=r"bert_sequence.txt",data_filename=  r"../datasets/ratings.csv"):
        print("="*10,"Reading .txt file with all item id and embeddings","="*10)
        with open(bert_filename) as f:
            reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
            feature_embed = {int(line[0]): np.array(list(map(float, line[1:])))
                    for line in reader}
        #print(feature_embed)
       # pd.DataFrame.from_dict(feature_embed)
        #pd.to_csv(r"C:\Users\7000030999\Desktop\feature_em.csv", index= True)
        data = pd.read_csv(r"/hdd/brijraj/interns/sonal/env/sars_tutorial_1_bert_plot_embeddings/recommenders/org10sec_session.csv")
        movie_ids = list(data['item_id'].unique())
        feature_embed_new = {key: feature_embed[key] for key in feature_embed if key in movie_ids}
        remove_list = pd.read_csv(r"/hdd/brijraj/interns/sonal/env/sars_tutorial_1_bert_plot_embeddings/recommenders/remove_list.csv")
        remove_list = list(remove_list['remove'])
       # print(remove_list)
        feature_embed_new2 = {key: feature_embed_new[key] for key in feature_embed_new if key in remove_list}
        pd.DataFrame.from_dict(feature_embed_new2)
        #print(len(feature_embed_new2))
        #print(feature_embed_new2)
        #pd.to_csv(r"C:\Users\7000030999\Desktop\feature_em_2.csv", index= True)
        return feature_embed_new2
    
    
    def fit(self, train_data):
        self.logger.info('Converting training data to GRU4Rec format')
        # parse training data to GRU4Rec format
        train_data = dataset_to_gru4rec_format(dataset=train_data)
        self.logger.info('Converting training data to GRU4Rec format complete')
        if not self.personalized:
            # fit GRU4Rec
            self.model = GRU4Rec(layers=self.session_layers,
                                 n_epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 learning_rate=self.learning_rate,
                                 momentum=self.momentum,
                                 dropout_p_hidden=self.dropout,
                                 session_key='session_id',
                                 item_key='item_id',
                                 time_key='ts',
                                embedding=768)
        else:
            if self.user_layers is None:
                raise ValueError('You should set the value of user_layers before training the personalized model.')

            if len(self.dropout) != 3:
                raise ValueError('dropout should be a 3 tuple with '
                                 '(user hidden, session hidden, user-to-session hidden) dropout values.')

            embed_dict = self.bert2dict("bert_embeddings_plot.txt",data_filename=  r"../datasets/ratings.csv")
            
            self.model = HGRU4Rec(session_layers=self.session_layers,
                                  user_layers=self.user_layers,
                                  batch_size=self.batch_size,
                                  n_epochs=self.epochs,
                                  learning_rate=self.learning_rate,
                                  momentum=self.momentum,
                                  dropout_p_hidden_usr=self.dropout[0],
                                  dropout_p_hidden_ses=self.dropout[1],
                                  dropout_p_init=self.dropout[2],
                                  session_key='session_id',
                                  user_key='user_id',
                                  item_key='item_id',
                                  time_key='ts',
                                  item_embedding = 768,
                                  init_item_embeddings = embed_dict
                                 )
        self.logger.info('Training started')
        self.model.fit(train_data)
        self.logger.info('Training completed')

    def recommend(self, user_profile, user_id=None):
        if not self.personalized:
            for item in user_profile:
                pred = self.model.predict_next_batch(np.array([self.pseudo_session_id]),
                                                     np.array([item]),
                                                     batch=1)
        else:
            if user_id is None:
                raise ValueError('user_id required by personalized models')
            for item in user_profile:
                pred = self.model.predict_next_batch(np.array([self.pseudo_session_id]),
                                                     np.array([item]),
                                                     np.array([user_id]),
                                                     batch=1)
        # sort items by predicted score
        pred.sort_values(0, ascending=False, inplace=True)
        # increase the psuedo-session id so that future call to recommend() won't be connected
        self.pseudo_session_id += 1
        # convert to the required output format
        return [([x.index], x._2) for x in pred.reset_index().itertuples()]
