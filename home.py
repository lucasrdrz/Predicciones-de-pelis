import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
import re
import math
import plotly.express as px
import streamlit as st

df = pd.read_csv('./datasets/df_completo.csv')

from sklearn.feature_extraction.text import TfidfVectorizer


tfidf = TfidfVectorizer(stop_words='english')

df['description'] = df['description'].fillna(' ')

tfidf_matrix = tfidf.fit_transform(df['description'])

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix)
indices = pd.Series(df.index,index=df['title']).drop_duplicates()

def get_recomenation(title,cosine_sim=cosine_sim):
    idx=indices[title]

    sim_score=list(enumerate(cosine_sim[idx]))

    sim_score = sorted(sim_score,key=lambda x: x[1],reverse=True)

    sim_score = sim_score[1:11]

    movie_indices = [i[0]for i in sim_score]

    return df['title'].iloc[movie_indices]


selected_title = st.selectbox("Seleccione una película:", df["title"])

# ejecutar la función get_recommendation() para el título seleccionado
recommendations = get_recomenation(selected_title)

# mostrar las recomendaciones en Streamlit
st.write("Películas recomendadas:")
for movie in recommendations:
    st.write(movie)