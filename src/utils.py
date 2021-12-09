"""
Pre- and postfilter items
"""

import numpy as np


def prefilter_items(data, n_popular=5000, item_features=None):
    fake_id = 999999

    # 1. Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'unique_buyers'}, inplace=True)

    # top_popular = popularity[popularity['unique_buyers'] > 0.4].item_id.tolist()

    # data = data[~data['item_id'].isin(top_popular)]
    # data.loc[data['item_id'].isin(top_popular)] = fake_id

    # 2. Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['unique_buyers'] < 0.02].item_id.tolist()

    data = data[~data['item_id'].isin(top_notpopular)]

    # 3. Уберем товары, которые не продавались за последние 12 месяцев
    ''' 12 месяцев = 52 недели '''

    # max_week = data['week_no'].max()
    # max_week_for_item = data_train.groupby('item_id')['week_no'].max().reset_index()

    '''Выделим товары, которые продавались последний раз менее года назад'''
    # new_items = max_week_for_item.loc[max_week_for_item['week_no'] > max_week - 52, 'item_id'].tolist()

    # data = data[data['item_id'].isin(new_items)]

    # 4. Уберем не интересные для рекоммендаций категории (department)

    #if item_features is not None:
    #    department_size = items_data.groupby('department')['item_id'].nunique().reset_index()
    #    department_size.columns = ['department', 'n_items']
    #    rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
    #    items_in_rare_departments = items_data[
    #        items_data['department'].isin(rare_departments)].item_id.unique().tolist()
    #    data = data[~data['item_id'].isin(items_in_rare_departments)]

    # 5. Уберем слишком дешевые и слишком дорогие товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.

    # Найдем среднюю цену за товар, т.к. клиенты могли покупать по разным ценам

    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    #data = data[data['price'] > 2]

    # Уберем слишком дорогие товарыs
    data = data[data['price'] < 50]

    # 6. Отфильтруем топ 5000
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top_5000 = popularity.sort_values('n_sold', ascending=False).head(n_popular).item_id.tolist()

    # заведем фиктивный товар - все товары, которые не попали в топ 5000, назовем 99999
    data.loc[~data['item_id'].isin(top_5000), 'item_id'] = fake_id

    return data


def postfilter_items(user_id, recommendations):
    pass