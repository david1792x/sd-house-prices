import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
import XGBoost

model = pickle.load(open('StreamlitApp/sd_pipeline.pkl', 'rb'))

st.set_page_config(layout='wide')
st.markdown('# San Diego County House Price Predictor :house_with_garden:')
      
st.markdown('## This app predicts the price of a house in San Diego County')
st.markdown('### Please enter the following information:')
st.markdown("""---""")
col1, col2, col3 = st.columns(3)
with col1:
    beds = st.select_slider('Number of bedrooms', options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    baths = st.select_slider('Number of bathrooms', options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    area = st.number_input('Square footage of the house', min_value = 500, max_value = 8000, value = 500, step = 250)
    home = st.selectbox('Home type', ('Single family', 'Condominium', 'Townhouse', 'Mobile home', 'Apartment'))

with col3:
    city = st.selectbox('City/Neighbourhood', ('Oceanside',
                                'Chula Vista',
                                'Escondido',
                                'El Cajon',
                                'Carlsbad',
                                'San Marcos',
                                'Vista',
                                'Santee',
                                'La Mesa',
                                'Spring Valley',
                                'La Jolla',
                                'Lakeside',
                                'Encinitas',
                                'Poway',
                                'Coronado',
                                'Rancho Santa Fe',
                                'National City',
                                'Lemon Grove',
                                'Imperial Beach',
                                'Del Mar',
                                'Solana Beach',
                                'Bonita',
                                'Cardiff',
                                'Jamul',
                                'Ramona',
                                'San Ysidro',
                                'Downtown',
                                'Clairemont',
                                'Point Loma',
                                'Mission Valley'))


lat_long = {'Oceanside': [-117.3110072008547, 33.215828846764346],
            'La Mesa': [-117.04050810250897, 32.77894560143369],
            'San Marcos': [-117.18052550394589, 33.13324804284103],
            'Mission Valley': [-117.12265038230883, 32.795264236881565],
            'Spring Valley': [-116.9920713265306, 32.72464886456401],
            'Escondido': [-117.08270729677913, 33.13200581825153],
            'Rancho Santa Fe': [-117.13905398635477, 33.017146914230025],
            'San Ysidro': [-117.05138629355608, 32.57372092840096],
            'El Cajon': [-116.93694464880383, 32.798379757894736],
            'Lakeside': [-116.92127663341647, 32.853978830423934],
            'Vista': [-117.23720125274724, 33.195082472527474],
            'Poway': [-117.08469116025303, 32.964638063784925],
            'Clairemont': [-117.19357497324417, 32.827602494983275],
            'Chula Vista': [-117.00319478425095, 32.627727637003844],
            'Jamul': [-116.87019878571428, 32.724345178571426],
            'Bonita': [-117.02167836792452, 32.664720283018866],
            'Solana Beach': [-117.25799776229509, 32.9918748852459],
            'La Jolla': [-117.25490830048078, 32.84048724759616],
            'Encinitas': [-117.268432888, 33.054620498666665],
            'Cardiff': [-117.27396172602741, 33.025002876712335],
            'Del Mar': [-117.22150338379531, 32.95269053091684],
            'Point Loma': [-117.23036406877578, 32.764945698762034],
            'Lemon Grove': [-117.03401306626506, 32.73061305421687],
            'Downtown': [-117.14120519426753, 32.72762529777069],
            'Carlsbad': [-117.28466555166217, 33.12165268823001],
            'National City': [-117.05926055970151, 32.6915661761194],
            'Santee': [-116.99045028524591, 32.8480271557377],
            'Coronado': [-117.1687791954023, 32.67705742528736],
            'Imperial Beach': [-117.1172809609375, 32.5788013046875],
            'Ramona': [-116.95042874999999, 33.018945575]}

home_dict = {'Single family': 'SINGLE_FAMILY',
             'Condominium': 'CONDO',
             'Townhouse': 'TOWNHOUSE',
             'Mobile home': 'MANUFACTURED',
             'Apartment': 'APARTMENT'}

def predict():
    home_type = home_dict[home]
    lat, long = lat_long[city][1], lat_long[city][0]
    data = np.array([[beds, baths, area, home_type, city, long, lat]])
    df = pd.DataFrame(data, columns = ['bedrooms', 'bathrooms', 'livingArea', 'homeType', 'city', 'longitude', 'latitude'])
    prediction = model.predict(df)
    return prediction

with col3:
    st.markdown("""---""")
    if st.button('Predict'):
        with st.container():
            st.markdown("""---""")
            st.write('Predicted price:')
            st.header(f'${predict().item():.0f}')