import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Market Basket Analysis Dashboard')

def load_data(file):
    try:
        df = pd.read_csv(file, header=0, skip_blank_lines=True)
        # Only use the first column, drop NaNs
        transactions = df.iloc[:, 0].dropna().apply(lambda x: [item.strip() for item in str(x).split(',') if item.strip()])
        return transactions
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.Series([])

uploaded_file = st.file_uploader('Upload your transactions CSV file', type=['csv'])
if uploaded_file:
    transactions = load_data(uploaded_file)
else:
    transactions = load_data('sample_transactions.csv')

st.write('### Sample Transactions')
st.write(transactions.head())

# Transaction encoding
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Frequent itemsets
min_support = st.slider('Minimum support', min_value=0.1, max_value=1.0, value=0.3, step=0.05)
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

st.write('### Frequent Itemsets')
st.dataframe(frequent_itemsets)

# Association rules
if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    st.write('### Association Rules')
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    # Heatmap of item correlations
    st.write('### Item Correlation Heatmap')
    corr = df_encoded.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
else:
    st.warning('No frequent itemsets found. Try lowering the minimum support.') 