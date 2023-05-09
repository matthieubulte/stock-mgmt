import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import io
from scipy.sparse import dok_matrix

def load_data(filename):
    SRC_ORDERS_ID = 'N° commande CN'
    SRC_ORDERS_ARTICLE = 'Article'
    SRC_ORDERS_QTY = 'Reste à livrer'

    SRC_STOCK_ID = 'Article'
    SRC_STOCK_QTY = 'Stock corrigé'

    o_orders_df = pd.read_excel(filename, sheet_name=0, skiprows=0)
    o_stock_df = pd.read_excel(filename, sheet_name=1, skiprows=0)
    
    orders_df = o_orders_df.rename({
        SRC_ORDERS_ID: 'order_id',
        SRC_ORDERS_ARTICLE: 'article_id',
        SRC_ORDERS_QTY: 'quantity'
    }, axis=1)[['order_id', 'article_id', 'quantity']]

    stock_df = o_stock_df.reset_index().rename({
        SRC_STOCK_ID: 'article_id',
        SRC_STOCK_QTY: 'stock'
    }, axis=1).groupby('article_id').sum(numeric_only=True).reset_index()

    # some stocks are negative, fix this
    stock_df.loc[stock_df['stock'] < 0, 'stock'] = 0

    tmp_order_df = pd.merge(orders_df, stock_df, on='article_id', how='left')
    missing_stock_ids = tmp_order_df[tmp_order_df.stock.isna()].article_id.unique()
    missing_stock_df = pd.DataFrame({'article_id': missing_stock_ids, 'stock': np.zeros_like(missing_stock_ids)})

    stock_df = pd.concat([stock_df, missing_stock_df]).reset_index(drop=True)
    stock_df['article_index'] = stock_df.index

    orders_df = pd.merge(orders_df, stock_df, on='article_id')

    order_id_to_idx = dict([ (oid, idx) for (idx, oid) in enumerate(orders_df['order_id'].unique()) ])
    orders_df['order_index'] = orders_df['order_id'].map(order_id_to_idx)

    orders_df['stock_available'] = orders_df['quantity'] <= orders_df['stock']
    order_to_available_df = orders_df.groupby('order_id')['stock_available'].all()
    orders_df =  pd.merge(orders_df, order_to_available_df, on='order_id', suffixes=('', '_order_total'))
    return orders_df, stock_df

def compute_optimal_alloc(orders_df, stock_df):
    def make_order_matrix(orders_df, num_orders, num_items):
        order_qty_matrix = dok_matrix((num_items, num_orders), dtype=np.float32)
        for _, order in orders_df.iterrows():
            order_qty_matrix[order['article_index'], order['order_index']] += order['quantity']
        return order_qty_matrix

    num_items = stock_df['article_index'].max()+1
    num_orders = orders_df['order_index'].max()+1

    order_qty_matrix = make_order_matrix(orders_df, num_orders, num_items)
    order_selection = cp.Variable(name='order_selection', shape=(num_orders,), boolean=True)
    order_vol = orders_df.groupby('order_index')['quantity'].sum().values


    prob = cp.Problem(cp.Maximize( order_vol.T @ order_selection ),
                    [order_qty_matrix @ order_selection <= stock_df['stock']])
    prob.solve()
    order_selection_sol = np.round(order_selection.value)

    return order_selection_sol, prob

def result_df(orders_df, order_selection_sol):
    out_df = orders_df.copy()
    out_df['should_deliver'] = out_df['order_index'].map(dict(enumerate(order_selection_sol)))
    return out_df[['order_id', 'should_deliver']].groupby(['order_id', 'should_deliver']).min().reset_index().sort_values(by='should_deliver', ascending=False).reset_index()

st.markdown(f"<h1 style='text-align: center'>Allocation de commandes</h1>", unsafe_allow_html=True)

in_file = st.file_uploader("Entrez le fichier de stocks et de commandes", type=["xlsx"])
if in_file is not None:
    print('File uploaded')
    orders_df, stock_df = load_data(in_file)
    
    print('Writing out ')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Commandes")
        st.write(orders_df[['order_id', 'article_id', 'quantity']])
        st.markdown(f'{orders_df["order_id"].nunique()} commandes uniques trouvées.')
        st.markdown(f'{orders_df.shape[0]} lignes de commandes trouvées.')

    with col2:
        st.markdown("### Stock")
        st.write(stock_df[['article_id', 'stock']])
        st.markdown(f'{stock_df["article_id"].nunique()} articles uniques en stock')
        st.markdown(f'Quantité totale: {np.round(stock_df["stock"].sum(), 2)}')

    print('Computing optimal alloc...', end='')
    order_selection_sol, prob = compute_optimal_alloc(orders_df, stock_df)

    print('done')

    st.markdown('## Allocation optimale') 
    st.markdown(f'Nombre de commandes satisfaites: {int(order_selection_sol.sum())} / {orders_df["order_id"].nunique()} (≈{ np.round(order_selection_sol.sum()/orders_df["order_id"].nunique()*100, 2) }% des commandes)')
    st.markdown(f'Nombre d\'articles correspondants: {int(prob.value)} (≈{np.round(prob.value/orders_df.quantity.sum()*100, 2)}% du volume commandé)')

    print('Making result_df')
    delivered_orders = result_df(orders_df, order_selection_sol)

    st.markdown('### Détails de l\'allocation optimale')
    st.write(delivered_orders)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        delivered_orders.to_excel(writer, index=False, sheet_name='Commandes')
        writer.save()
        
        st.download_button(
            label="Télécharger",
            data=buffer,
            file_name=f'Resultat {in_file.name}',
            mime="application/vnd.ms-excel"
        )
