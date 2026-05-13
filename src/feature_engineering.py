import pandas as pd
import numpy as np
import inflection


def transform_data(df: pd.DataFrame) -> pd.DataFrame:

    # ==============================
    # RENAME COLUMNS
    # ==============================
    cols_old = df.columns
    cols_new = list(map(lambda x: inflection.underscore(x), cols_old))
    df.columns = cols_new

    # ==============================
    # CLEANING
    # ==============================
    df = df.dropna(subset=['customer_id']).copy()
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d-%b-%y')
    df['customer_id'] = df['customer_id'].astype(int)

    df = df[df['unit_price'] > 0].copy()
    df = df[df['stock_code'].str.match(r'^\d+$', na=False)].copy()

    # ==============================
    # SPLIT PURCHASE / RETURNS
    # ==============================
    df_purchase = df[~df['invoice_no'].str.startswith('C')].copy()
    df_returns = df[df['invoice_no'].str.startswith('C')].copy()

    # ==============================
    # BASE CUSTOMER TABLE
    # ==============================
    df_ref = (
        df.drop(['invoice_no', 'stock_code', 'description',
                 'quantity', 'invoice_date', 'unit_price', 'country'], axis=1)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ==============================
    # GROSS REVENUE
    # ==============================
    df_purchase['gross_revenue'] = df_purchase['quantity'] * df_purchase['unit_price']

    df_monetary = (
        df_purchase.groupby('customer_id')['gross_revenue']
        .sum()
        .reset_index()
    )

    df_ref = df_ref.merge(df_monetary, on='customer_id', how='left').dropna()

    # ==============================
    # RECENCY
    # ==============================
    df_recency = (
        df_purchase.groupby('customer_id')['invoice_date']
        .max()
        .reset_index()
    )

    df_recency['recency_days'] = (
        df_purchase['invoice_date'].max() - df_recency['invoice_date']
    ).dt.days

    df_ref = df_ref.merge(df_recency[['customer_id', 'recency_days']], on='customer_id', how='left')

    # ==============================
    # FREQUENCY
    # ==============================
    df_freq = (
        df_purchase.groupby('customer_id')['invoice_no']
        .nunique()
        .reset_index()
        .rename(columns={'invoice_no': 'frequency'})
    )

    df_ref = df_ref.merge(df_freq, on='customer_id', how='left')

    # ==============================
    # AVG TICKET
    # ==============================
    df_order_value = (
        df_purchase.groupby(['customer_id', 'invoice_no'])['gross_revenue']
        .sum()
        .reset_index()
    )

    df_avg_ticket = (
        df_order_value.groupby('customer_id')['gross_revenue']
        .mean()
        .reset_index()
        .rename(columns={'gross_revenue': 'avg_ticket'})
    )

    df_ref = df_ref.merge(df_avg_ticket, on='customer_id', how='left')

    # ==============================
    # LIFETIME
    # ==============================
    df_lifetime = (
        df_purchase.groupby('customer_id')['invoice_date']
        .agg(['min', 'max'])
        .reset_index()
    )

    df_lifetime['customer_lifetime_days'] = (
        df_lifetime['max'] - df_lifetime['min']
    ).dt.days

    df_ref = df_ref.merge(
        df_lifetime[['customer_id', 'customer_lifetime_days']],
        on='customer_id',
        how='left'
    )

    # ==============================
    # VELOCITIES
    # ==============================
    df_ref['revenue_velocity'] = df_ref['gross_revenue'] / np.maximum(df_ref['customer_lifetime_days'], 1)

    df_total_items = (
        df_purchase.groupby('customer_id')['quantity']
        .sum()
        .reset_index()
        .rename(columns={'quantity': 'total_items'})
    )

    df_ref = df_ref.merge(df_total_items, on='customer_id', how='left')

    df_ref['items_velocity'] = df_ref['total_items'] / (df_ref['customer_lifetime_days'] + 30)

    # ==============================
    # BASKET SIZE
    # ==============================
    df_items_order = (
        df_purchase.groupby(['customer_id', 'invoice_no'])['quantity']
        .sum()
        .reset_index()
    )

    df_basket_size = (
        df_items_order.groupby('customer_id')['quantity']
        .mean()
        .reset_index()
        .rename(columns={'quantity': 'basket_size'})
    )

    df_ref = df_ref.merge(df_basket_size, on='customer_id', how='left')

    # ==============================
    # PRODUCT FEATURES
    # ==============================
    df_unique_products = (
        df_purchase.groupby('customer_id')['stock_code']
        .nunique()
        .reset_index()
        .rename(columns={'stock_code': 'unique_products'})
    )

    df_ref = df_ref.merge(df_unique_products, on='customer_id', how='left')

    df_ref['product_loyalty'] = df_ref['total_items'] / df_ref['unique_products']
    df_ref['product_loyalty'] = df_ref['product_loyalty'].replace([np.inf, -np.inf], 0).fillna(0)

    # ==============================
    # AVG RECENCY
    # ==============================
    df_purchase_sorted = df_purchase.sort_values(['customer_id', 'invoice_date'])
    df_purchase_sorted['days_between'] = df_purchase_sorted.groupby('customer_id')['invoice_date'].diff().dt.days

    df_avg_recency = (
        df_purchase_sorted.groupby('customer_id')['days_between']
        .mean()
        .reset_index()
        .rename(columns={'days_between': 'avg_recency_days'})
    )

    df_ref = df_ref.merge(df_avg_recency, on='customer_id', how='left')
    df_ref['avg_recency_days'] = df_ref['avg_recency_days'].fillna(df_ref['customer_lifetime_days'])

    # ==============================
    # RETURNS
    # ==============================
    df_returns['return_revenue'] = df_returns['quantity'] * df_returns['unit_price']

    df_return_value = (
        df_returns.groupby('customer_id')['return_revenue']
        .sum()
        .abs()
        .reset_index()
        .rename(columns={'return_revenue': 'return_value'})
    )

    df_return_orders = (
        df_returns.groupby('customer_id')['invoice_no']
        .nunique()
        .reset_index()
        .rename(columns={'invoice_no': 'return_orders'})
    )

    df_returns_features = df_return_value.merge(df_return_orders, on='customer_id', how='outer')

    df_ref = df_ref.merge(df_returns_features, on='customer_id', how='left')
    df_ref[['return_value', 'return_orders']] = df_ref[['return_value', 'return_orders']].fillna(0)

    df_ref['return_rate'] = df_ref['return_orders'] / (df_ref['return_orders'] + df_ref['frequency'])

    df_ref['return_value_ratio'] = np.where(
        df_ref['gross_revenue'] > 0,
        df_ref['return_value'] / df_ref['gross_revenue'],
        0
    ).clip(0, 1)

    df_ref['net_revenue'] = df_ref['gross_revenue'] - df_ref['return_value']

    df_ref = df_ref[~((df_ref['net_revenue'] <= 0) | (df_ref['return_value_ratio'] >= 0.95))]

    return df_ref