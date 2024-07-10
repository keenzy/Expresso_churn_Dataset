import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.model_selection import train_test_split

# give a title to our app 
st.title('Expresso churn Dataset') 
# Export data
st.header('Exploration des données du dataset')
data=pd.read_csv('Expresso_churn_dataset.csv')
st.dataframe(data)

st.subheader('Rapport d’analyse exploratoire des données')

#profile = ProfileReport(data)
#st_profile_report(profile)
st.subheader('Gestion données manquantes')
st.write('Nombre de données manquantes')
st.warning('data.isnull().sum()')
st.write(data.isnull().sum())
st.write('Suppresion des données manquantes')
st.warning('data.dropna(inplace=True)')
data.dropna(inplace=True)
st.warning('data.isnull().sum()')
st.write(data.isnull().sum())
st.write('Gestion des valeurs dupliquées')
st.warning('data.duplicated().sum()')
st.write(data.duplicated().sum())

#Visualisation valeurs abberantes avec plotly boite à moustaches
st.subheader('Visualisation des valeurs abberantes avec box plots')

fig1 = px.box(data, y='MONTANT')
fig2 = px.box(data, y='REVENUE')
fig3 = px.box(data, y='FREQUENCE')
fig4 = px.box(data, y='DATA_VOLUME')
fig = make_subplots(rows=1, cols=4, subplot_titles=['Montant', 'REVENUE', 'FREQUENCE', 'DATA VOLUME'])


fig.add_trace(fig1.data[0], row=1, col=1)
fig.add_trace(fig2.data[0], row=1, col=2)
fig.add_trace(fig3.data[0], row=1, col=3)
fig.add_trace(fig4.data[0], row=1, col=4)
st.plotly_chart(fig, use_container_width=True)

#Visualisation valeurs abberantes avec nuages de points
st.subheader('Visualisation des valeurs abberantes avec les nuages de points')
           
fig1 = px.scatter(data, x="MONTANT", y="REVENUE",
                 size="FREQUENCE", 
                )
st.plotly_chart(fig1, use_container_width=True)

st.write('En observant les boites à moustaches isolément, on pourrait croire que le dataset contient des données abbérantes, mais les nuages de points démontrent que le revenu dépend du montant et de la fréquence de charge de chaque client')

st.subheader('Encodage données catégorielles')
#ENCODER DONNES CATEGORIELLES
data['REGION'] = data['REGION'].astype('category').cat.codes
data['TENURE'] = data['TENURE'].astype('category').cat.codes
data['MRG'] = data['MRG'].astype('category').cat.codes
data['TOP_PACK'] = data['TOP_PACK'].astype('category').cat.codes


st.dataframe(data)



#Entrainez les données
X = data[['REGULARITY','REGION','FREQUENCE_RECH','TENURE']]
y = data['CHURN']
# We split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
st.write(X_test.head())