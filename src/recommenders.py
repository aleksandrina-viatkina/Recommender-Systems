import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True):

        # Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать

        self.user_item_matrix = self.prepare_matrix(data)  
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T
        
        self.sparse_user_item = csr_matrix(self.user_item_matrix).tocsr()
        
        self.model = self.fit()
        self.own_recommender = self.fit_own_recommender()

    @staticmethod
    def prepare_matrix(data, fake_id=999999):
   
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)

        # Взвешиваем по доле продаж в общем объеме
        sales = data.groupby('item_id')['sales_value'].sum().reset_index()
        sales.rename(columns={'sales_value': 'sold'}, inplace=True)
        total_sum = np.sum(sales['sold']) - sales.loc[sales['item_id'] == fake_id, 'sold'].item()
        sales['weight'] = sales['sold'] / total_sum

        for column in user_item_matrix.columns.to_list():
            user_item_matrix.loc[:, column] *= sales.loc[sales['item_id'] == column, 'weight'].item()

        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    def fit_own_recommender(self):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K = 1, num_threads = 4)
        own_recommender.fit(csr_matrix(self.user_item_matrix).T.tocsr())

        return own_recommender
    
    def get_own_recommendations(self, user, n=5, fake_id=999999):
        """Рекомендуем топ-N товаров"""

        res = [self.id_to_itemid[rec[0]] for rec in
               self.own_recommender.recommend(userid=self.userid_to_id[user],
                                              user_items=self.sparse_user_item,
                                              N=n,
                                              filter_already_liked_items=False,
                                              filter_items=[self.itemid_to_id[fake_id]],
                                              recalculate_user=True)]
        return res

    def fit(self, n_factors=35, regularization=0.1, iterations=10, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(self.user_item_matrix).T.tocsr())

        return model

    def get_model_recommendations(self, user, n=5, fake_id=999999):

        recs = [self.id_to_itemid[rec[0]] for rec in
                self.model.recommend(userid=self.userid_to_id[user],
                                     user_items=self.sparse_user_item,
                                     N=n,
                                     filter_already_liked_items=False,
                                     filter_items=[self.itemid_to_id[fake_id]],  # сразу исключаем искуственный id
                                     recalculate_user=False)]
        return recs

    def get_similar_items_recommendation(self, user, n=5, fake_id = 999999):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
    
        res = []
        # получим рекомендации для конкретного юзера 
       
        recs = self.get_model_recommendations(user, n)
        for rec in recs:
            similar_items = self.model.similar_items(self.itemid_to_id[rec], N=3)
            i=1
            item_id = self.id_to_itemid[similar_items[1][0]]
            # Исключаем фиктивный id и дубликаты
            while item_id == fake_id or item_id in res:
                item_id = self.id_to_itemid[similar_items[i+1][0]]
            res.append(item_id)

        assert len(res) == n, 'Количество рекомендаций != {}'.format(n)
        return res
        
        
    def get_similar_users_recommendation(self, user, n=5, fake_id = 999999):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        res = []

        similar_userids = self.model.similar_users(self.userid_to_id[user], N=n + 1)
        
        #Будем добавлять так, чтобы не было дубликатов
        for userid in similar_userids[1:]:
            user = self.id_to_userid[userid[0]]
            recs = self.get_model_recommendations(user, n=5)
            i=0
            while recs[i] in res or recs[i] == fake_id:
                i+=1
            res.append(recs[i])

        assert len(res) == n, 'Количество рекомендаций != {}'.format(n)
        return res

    
