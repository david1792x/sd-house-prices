# Deploying a web application for predictions using Streamlit

As of now, we have **retrieved** our data from Zillow, **cleaned** and **visualized** it, and created a **machine learning** model that predicts house price based on the data. The final step is to create a very simple **web app** that allows any user to input house data and get a **prediction** using the trained model. We will construct this app using `Streamlit`, which is a very **user friendly** environment for creating **data-centric** web apps in Python.

The directory where the **.py** file associated with the web app, the **model** and the **requirements** of the app are stored, is located [here](StreamlitApp/).
The app was deployed using **Streamlit Cloud**, and it can be accessed using [this link](https://david1792x-sd-house-prices.streamlit.app/).

When the **web app** is opened, the following menu is, where the user can select between the different **parameters** that compose our model. 

<div align = 'center'>
  
| <img src='/images/streamlit_1.png' height="400">         |
|:-------------------------------------------------:|
| ***Figure 1.**  Streamlit web application*               |
  
</div>

Once the user **inputs** the house data, a **prediction** is made when clicking the predict button. In this case, the model predicts a price of **\$903,068** for a **3 bedroom, 3 bathroom condominium** with a **living area** of **1250 sqft**, located in **Carlsbad, CA**.

<div align = 'center'>
  
| <img src='/images/streamlit_2.PNG' height="400">         |
|:-------------------------------------------------:|
| ***Figure 2.**  Prediction in web application*               |
  
</div>

Something that must be noted regarding this project is that the model was trained on **real data** from the **real estate market**. The model was not trained to predict prices of **unlikely parameters**, like a house with **10** bedrooms and **1** bathroom, or a **7000 sqft** home with **1** bedroom. Even if the web app allows these kinds of homes to be inputted, the predictions are **not accurate** at all and no prediction made with this web app should be taken as a given, since it is a very basic model with a lot of room for **improvement**. 
