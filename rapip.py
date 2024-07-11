import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Data preparation
data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    'lagu': [
        "The Circus Is Leaving Town,Dusty Wreath,It's Hard To Kill A Bad Thing",
        "The Circus Is Leaving Town,Dusty Wreath,It's Hard To Kill A Bad Thing",
        "The Circus Is Leaving Town,Dusty Wreath,It's Hard To Kill A Bad Thing",
        "The Circus Is Leaving Town,Honey Child What Can I Do?,(Do You Wanna) Come Walk With Me?",
        "The Circus Is Leaving Town,Honey Child What Can I Do?,(Do You Wanna) Come Walk With Me?",
        "The Circus Is Leaving Town,Honey Child What Can I Do?,(Do You Wanna) Come Walk With Me?",
        "The Circus Is Leaving Town,Honey Child What Can I Do?,(Do You Wanna) Come Walk With Me?",
        "The Circus Is Leaving Town,Revolver,The False Husband",
        "The Circus Is Leaving Town,Revolver,The False Husband",
        "The Circus Is Leaving Town,Revolver,The False Husband",
        "Dusty Wreath,It's Hard To Kill A Bad Thing,Black Mountain",
        "Dusty Wreath,It's Hard To Kill A Bad Thing,Black Mountain",
        "Dusty Wreath,It's Hard To Kill A Bad Thing,Black Mountain",
        "Dusty Wreath,It's Hard To Kill A Bad Thing,Black Mountain",
        "Saturday's Gone,Sandhills - Lockdown Sessions,Late Afternoon - Lockdown Sessions",
        "Saturday's Gone,Sandhills - Lockdown Sessions,Late Afternoon - Lockdown Sessions",
        "Saturday's Gone,Sandhills - Lockdown Sessions,Late Afternoon - Lockdown Sessions",
        "Saturday's Gone,Sandhills - Lockdown Sessions,Late Afternoon - Lockdown Sessions",
        "The Circus Is Leaving Town,Sandhills - Lockdown Sessions,Late Afternoon - Lockdown Sessions",
        "Black Mountain, Honey Child What Can I Do?,Dusty Wreath",
        "Feel It Still,The Circus Is Leaving Town,Sandhills - Lockdown Sessions",
        "Reality (Radio Edit) (feat. Janieck Devy),Dusty Wreath,Rebecca You - Lockdown Sessions",
        "Prayer In C (Robin Schulz Radio Edit),Honey Child What Can I Do?,Just Like Tears In The Morning - Lockdown Sessions",
        "Mine,It's Hard To Kill A Bad Thing,Bill McCai - Lockdown Sessions",
        "Toosie Slide,Saturday's Gone,Shadows Fall - Lockdown Sessions",
        "Abandoned Castle ~ Curse of Darkness ~,(Do You Wanna) Come Walk With Me?,Walking In The Winter - Lockdown Sessions",
        "Blue Serenade,Ramblin' Man,Late Afternoon - Lockdown Sessions",
        "Bloody Tears,Revolver,Liezah - Lockdown Sessions",
        "Vampire Killer,Ballad Of The Broken Seas,Perimeter",
        "Darkness of Fear,The False Husband,Perimeter - Shorter"
    ]
}

# Mengatur judul tab
st.set_page_config(page_title="Apriori Kelompok 11")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Create DataFrame
df = pd.DataFrame(data)

# Split the 'lagu' column into multiple columns
df_expanded = df['lagu'].str.split(',', expand=True)

# Create a one-hot encoded DataFrame
df_onehot = pd.get_dummies(df_expanded.stack()).groupby(level=0).sum()

st.title('Analisis Asosiasi Preferensi Lagu dengan Algoritma Apriori')
st.write("")
st.write("")
st.write('Aplikasi ini melakukan analisis asosiasi preferensi lagu dengan menggunakan algoritma Apriori.')
st.write('Nama Anggota Kelompok:')
st.write('1. Alfa Rado Andre Yusa Saka Tory (210103086)')
st.write('2. Muhammad Syaifullah (210103109)')
st.write('3. Rafif Rizqy Alfiansyah (210103114)')

st.subheader('Memuat Data')
st.write('Dataset yang digunakan:')
st.write(df)

st.subheader('Memproses Data')
st.write('Data setelah di-expand:')
st.write(df_expanded)
st.write('Data Tabular:')
st.write(df_onehot)

st.subheader('Menerapkan Algoritma Apriori')

# User input for support and confidence
min_support = st.slider('Minimum Support', 0.0, 1.0, 0.2, 0.01)
min_confidence = st.slider('Minimum Confidence', 0.0, 1.0, 0.6, 0.01)

st.write(f'Apriori Dengan Nilai Support = {min_support} dan Nilai Confidence = {min_confidence}')

# Apply Apriori algorithm
frequent_itemsets = apriori(df_onehot, min_support=min_support, use_colnames=True)
st.write('Frequent Itemsets:')
st.write(frequent_itemsets)

st.subheader(' Membuat Association Rules')
# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
st.write('Association Rules:')
st.write(rules)

st.subheader('Hasil')
st.write('Menampilkan aturan asosiasi yang dihasilkan:')
st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
