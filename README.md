# **House Price Predictions in San Diego County**

San Diego County is located in southwest California. It is one of the most populous counties in the United States, with over 3.3 million inhabitants. San Diego County has one of the most expensive real estate markets in the country, caused by a rising millenial population looking for homes, and a stable, diverse economy fueled by industries like healthcare, technology, defense, international trade and scientific reseach, creating a big job market that increases demand for housing. The geographical location of San Diego County is a big factor in the power of the real estate market, since it is located next to the Pacific Ocean and has a really comfortable weather all year long.

In this project, we will webscrape the **Zillow** website to extract a dataset of real estate listings in the San Diego County, explore the statistics and patterns behind the scraped dataset, and build a robust machine learning pipeline that cleans and prepares our data, and estimates a sale price based on key predictor variables like living area, neighborhood, number of bedrooms, among others, with the objective of making a simple web application that inputs the information of a house in San Diego County and returns an estimated selling price.

Zillow offers an estimate of home prices called **Zestimate** that is based on public real estate transaction data, home information, tax assessment information and similar homes sold in the area. This model offers a stable prediction for house price that has a median error rate of 3.2% for in market homes and 7.5% for off market homes, which is what the proposed model is trained on in order to mitigate the effects of errors in reported selling prices and to leverage the advanced prediction algorithm to make a much simpler model with good results.

