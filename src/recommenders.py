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

    def __init__(self, data, weighting=True, show_progress=False):

        self.user_item_matrix = data  # pd.DataFrame
        self.user_item_matrix = self.user_item_matrix.astype(float)

        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(
            self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix, show_progress=show_progress)

        binary_user_item_matrix = data.copy()
        binary_user_item_matrix[binary_user_item_matrix > 0] = 1.
        binary_user_item_matrix = binary_user_item_matrix.astype(float)
        self.own_recommender = self.fit_own_recommender(binary_user_item_matrix, show_progress=show_progress)

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

    @staticmethod
    def fit_own_recommender(user_item_matrix, show_progress=False):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=show_progress)

        return own_recommender

    def get_own_recommends(self, user_id, n_recommends=5, filter_999999=True):
        recs = self.own_recommender.recommend(userid=self.userid_to_id[user_id],  # userid - id от 0 до N
                                              user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                              # на вход user-item matrix
                                              N=n_recommends,  # кол-во рекомендаций
                                              filter_already_liked_items=False,
                                              filter_items=[self.itemid_to_id[999999]] if filter_999999 else None,
                                              recalculate_user=True)

        recs = [self.id_to_itemid[rec[0]] for rec in recs]

        return recs

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4, show_progress=False):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=show_progress)

        return model

    def get_similar_items_recommendation(self, user_id, N=5, filter_999999=True):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        own_recs = self.own_recommender.recommend(userid=self.userid_to_id[user_id],  # userid - id от 0 до N
                                                  user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                                  # на вход user-item matrix
                                                  N=N,  # кол-во рекомендаций
                                                  filter_already_liked_items=False,
                                                  filter_items=[self.itemid_to_id[999999]] if filter_999999 else None,
                                                  recalculate_user=True)

        # Для каждой найденной собственной рекомендации ищем максиально похожий товар
        # Если найденный товар уже есть в списке рекомендаций, ищем следующий
        recs = []
        for own_rec in own_recs:
            N_neigbors = 2
            success = False
            while not success:
                rec = self.model.similar_items(own_rec[0], N=N_neigbors)
                rec = self.id_to_itemid[rec[-1][0]]
                if rec not in recs:
                    recs.append(rec)
                    success = True
                else:
                    N_neigbors += 1

        if len(recs) != 5:
            print('Error: Рекоммендаций меньше, чем ожидается. user_id:', user_id)
        # assert len(recs) == N, 'Количество рекомендаций != {}'.format(N)
        return recs

    def get_similar_users_recommendation(self, user_id, N=5, filter_999999=True):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        users = self.model.similar_users(userid=self.userid_to_id[user_id], N=N)

        max_iterations = 10
        recs = []
        # Находим N похожих пользователей и берем самый популярный товар от каждого пользователя
        # Если найденный товар уже есть в списке рекомендаций, ищем следующий популярный у пользователя
        for user in users:
            N_recommends = 1
            success = False

            iterations = 0
            while not success:
                iterations += 1
                rec = self.own_recommender.recommend(userid=user[0],  # userid - id от 0 до N
                                                     user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                                     # на вход user-item matrix
                                                     N=1,  # кол-во рекомендаций
                                                     filter_already_liked_items=False,
                                                     filter_items=[
                                                         self.itemid_to_id[999999]] if filter_999999 else None,
                                                     recalculate_user=True)

                if len(rec) == 0 or iterations >= max_iterations:
                    success = True
                    break

                rec = self.id_to_itemid[rec[-1][0]]
                if rec not in recs:
                    recs.append(rec)
                    success = True
                else:
                    N_recommends += 1

        # assert len(recs) == N, 'Количество рекомендаций != {}'.format(N)
        if len(recs) != 5:
            print('Error: Рекоммендаций меньше, чем ожидается. user_id:', user_id)
        return recs