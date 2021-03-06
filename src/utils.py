def prefilter_items(data,
                    filter_top_popular=None,
                    exclude_top_popular=None,
                    exclude_top_unpopular=None,
                    exclude_items=None,
                    exclude_not_sold_weeks=None,
                    exclude_cheaper_than=None,
                    exclude_more_expensive_than=None
                    ):
    data_filtered = data.copy()

    # Уберем самые популярные
    if exclude_top_popular is not None:
        popularity = data_filtered.groupby('item_id')['quantity'].sum().reset_index()
        popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
        top_N = popularity.sort_values('n_sold', ascending=False).head(exclude_top_popular).item_id.tolist()
        data_filtered.loc[data_filtered['item_id'].isin(top_N), 'item_id'] = 999999

    # Уберем самые непопулряные
    if exclude_top_unpopular is not None:
        unpopularity = data_filtered.groupby('item_id')['quantity'].sum().reset_index()
        unpopularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
        untop_N = unpopularity.sort_values('n_sold', ascending=True).head(exclude_top_unpopular).item_id.tolist()
        data_filtered.loc[data_filtered['item_id'].isin(untop_N), 'item_id'] = 999999

    # Уберем товары, которые не продавались за последние 12 месяцев
    if exclude_not_sold_weeks is not None:
        week_no_treshold = data_filtered['week_no'].max() - exclude_not_sold_weeks
        sold_items = data_filtered.loc[data_filtered['week_no'] >= week_no_treshold, 'item_id'].tolist()
        sold_items = list(set(sold_items))
        data_filtered.loc[~data_filtered['item_id'].isin(sold_items), 'item_id'] = 999999

    # Уберем не интересные для рекоммендаций товары
    if exclude_items is not None:
        data_filtered.loc[data_filtered['item_id'].isin(exclude_items), 'item_id'] = 999999

        # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    if exclude_cheaper_than is not None:

        def filter_cheaper(row, max_price):
            if row['quantity'] == 0:
                return row['item_id']
            elif row['sales_value'] / row['quantity'] > max_price:
                return row['item_id']
            else:
                999999

        data_filtered['item_id'] = data_train.apply(lambda row: filter_cheaper(row, exclude_cheaper_than), axis=1)

    # Уберем слишком дорогие товары
    if exclude_more_expensive_than is not None:

        def filter_more_expensive(row, min_price):
            if row['quantity'] == 0:
                return row['item_id']
            elif row['sales_value'] / row['quantity'] < min_price:
                return row['item_id']
            else:
                999999

        data_filtered['item_id'] = data_train.apply(lambda row: filter_more_expensive(row, exclude_more_expensive_than),
                                                    axis=1)

    # Оставим только 5000 самых популярных товаров
    if filter_top_popular is not None:
        popularity = data_filtered.query('item_id != 999999').groupby('item_id')['quantity'].sum().reset_index()
        popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
        top_N = popularity.sort_values('n_sold', ascending=False).head(filter_top_popular).item_id.tolist()
        # добавим, чтобы не потерять юзеров
        data_filtered.loc[~data_filtered['item_id'].isin(top_N), 'item_id'] = 999999

    return data_filtered


def postfilter_items():
    pass