# **House Price Predictions in San Diego County** :house_with_garden: :moneybag:

## **Introduction**
**San Diego County** is located in southwest California. It is one of the most populous counties in the United States, with over 3.3 million inhabitants, comprising 18 different cities, including the San Diego - Chula Vista - Carlsbad Metropolitan Area, which is the 17th most populous metropolitan area in the United States. San Diego County also has a direct border with Mexico, and the San Ysidro Port of Entry is the busiest land border crossing in the entire western hemisphere.

<div align="center">

|  <img src='images/mapimage.jpeg' width="575">                     |<img src='images/county_location_2.png' height="438">                              |
|:----------------------------------------------------------------------:|:---------------------------------------------------------------------------------:|
| ***Figure 1.**  San Diego County in Google Maps*                        | ***Figure 2.** Location of San Diego County*                        |

</div>

San Diego County has one of the most expensive real estate markets in the country, caused by a rising millenial population looking for homes, and a stable, diverse economy fueled by industries like tourism, healthcare, technology, defense, government dependencies, international trade and scientific reseach, creating a big job market that increases demand for housing. Like most US counties, the San Diego County urban layout follows a trend of big commercial and industrial areas, surrounded by beautiful, calm **suburbs** that are mainly residential.

<div align="center"> 

| <img src='images/suburb.jpg' height="300">                       |<img src='images/sd_home.jpg' height="300">                       |
|:----------------------------------------------------------------------:|:---------------------------------------------------------------------------------:|
| ***Figure 3.**  Black Mountain area in San Diego County*                        | ***Figure 4.** Typical suburban home in San Diego County*          |

</div> 

The geographical location of San Diego County is a big factor in the power of the real estate market, since it is located next to the Pacific Ocean and has a really **comfortable weather** and **nice views** all year long, with a big percent of its residents being senior citizens that look for a beautiful and calm city to live in.

<div align="center"> 

| <img src='images/sd_highrise.png' height="300">                      |<img src='images/la_jolla.png' height="300">                   |
|:----------------------------------------------------------------------:|:---------------------------------------------------------------------------------:|
| ***Figure 5.**  Downtown San Diego area*                        | ***Figure 6.** La Jolla area*          |

</div> 

## **Project overview**
In this project, we will webscrape the **Zillow** website to extract a dataset of real estate listings in the San Diego County, explore the **statistics** and **patterns** behind the scraped dataset, and build a robust **machine learning** pipeline that cleans and prepares our data, and estimates a **sale price** based on key predictor variables like living area, neighborhood, number of bedrooms, among others, with the objective of making a simple **web application** that inputs the information of a house in San Diego County and returns an estimated selling price.

The general workflow diagram of the project is shown below. 

<div align="center">
    
|<img src='images/ml_diagram.jpg' width = '750'>|
|:---------------------------------------------:|
| ***Figure 7.** Workflow diagram of project*   |
    
</div>

## **Objectives**
- Use **Zillow** house listings to create a **dataset** of homes sold in San Diego county between April 2022 and April 2023
- Create **visualizations** and extract **patterns** from the data
- Train a simple **machine learning** model to **predict** house prices based on home information
- Design and deploy a **web application** that predicts house prices according to user input

## **Project development**
### Part I. Extracting dataset from Zillow
The first step in developing the project was to obtain a **dataset**. Zillow was chosen as the website to extract the data from, since it is the largest real estate website in the United States with over **135 million** real estate listings. A fetch request is used to access Zillow's `GetSearchPageState` API call, which contains information of house listings that match a specific query. Then, the request responses are combined to make a .csv file of over 26,000 listings of houses sold between April 2022 and April 2023 in San Diego County.

<br/>

<div align='center'>
    
|Detailed information|
|:---------------------------------------------:|
|Link to the [markdown](markdown/extracting_dataset.md) file|
| Link to the [Jupyter](jupyter/extracting_dataset.ipynb) Notebook   |

</div>

---

### Part II. Exploring and cleaning our dataset
After obtaining the dataset, some **cleaning** needs to be done, since it contains the raw data from Zillow's listings, which often times is filled with incorrect values. After cleaning the data from **outliers** and **errors**, we can show some **trends** in graphic form, like price histograms, relationship of the different variables with house price, prices of different zones in the map, among others.

<br/>

<div align='center'>
    
|Detailed information|
|:---------------------------------------------:|
|Link to the [markdown](markdown/exploratory_data_analysis.md) file|
|Link to the [Jupyter](jupyter/exploratory_data_analysis.ipynb) Notebook |

</div>

---

### Part III. Building a machine learning model
Once the data is cleaned, we establish a **machine learning** pipeline to create a model that is capable of learning the patterns in the data in order to make **predictions** on house price. The pipeline incorporates a preprocessor that transforms our data to an algorithm-friendly structure, and an extreme gradient boosting regression tree estimator using the `XGBoost` module that is trained and tuned until the performance metrics are good enough for our application purpose.

<br/>

<div align='center'>
    
|Detailed information|
|:---------------------------------------------:|
|Link to the [markdown](markdown/ml_modeling.md) file|
|Link to the [Jupyter](jupyter/ml_modeling.ipynb) Notebook|

</div>

---

### Part IV. Deploying a web application
Now that we have a trained model that performs as desired, we can save it and use it to build a **web application** that lets the user input home information, and returns an estimated price for the described home. The framework used to develop the web application is `Streamlit`, and it was chosen because of its simplicity and **data-centered** approach. Finally, the web app is deployed using **Streamlit Cloud Services** and open for anyone's use.  

<br/>

<div align='center'>
    
|Detailed information|
|:---------------------------------------------:|
|Link to the [markdown](markdown/deploy_web_app.md) file|
| Link to the [Python](StreamlitApp/predictor_app.py) script|
|Link to the [web application](https://david1792x-sd-house-prices.streamlit.app/)|

</div>

---

## **Discussion and final remarks**

Link to the complete [Jupyter](jupyter/sd_house_prices_complete.ipynb) Notebook of the project (not descriptive).
