
# **Analysis of Movie Data**

<center><img src="https://media.licdn.com/dms/image/v2/D4D12AQE5GUXEwi8Pwg/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1679549353292?e=1733356800&v=beta&t=F1GWxuBky4CRYs8xitF7UdVyZ4e-XL6YruE-0CXR1nA" width="500" height="292"/></center>

## Overview

The purpose of this project is to use multiple datasets containing movie information to identify what movie-related factors lead to success as quantified by return on investment (ROI). The analysis aims to provide insights that could contribute to a film studio or production company's decision-making regarding the types of movies to produce. Ordinary Least Squares (OLS) regression is used with ROI as the dependent variable to estimate how effective selected independent variables such as film genre or film runtime are at predicting movie success. Due to the nature of the datasets I elected to use this analysis is set in 2019.

### Business Understanding

Having seen how big companies especially in tech (e.g. Amazon, Apple, and Netflix) have started investing heavily in original video content production, our company is now looking for opportunities in the film industry. I have been tasked with exploring what types of films are currently doing the best at the box office. 

The film industry has seen steady growth over this decade and is projected healthy growth going forward thanks to the key growth drivers such as the rise of streaming platforms and growing entertainment demands in international markets like China. 

<img src="https://github.com/user-attachments/assets/af088291-c09a-4ff5-8609-7d01399626e5" width="600" height="480.09">

There are several market trends of note currently which are all undergirded by the globalization of content. The aforementioned rise of streaming platforms and direct-to-consumer models of original video content is one. Streaming is already a dominant force in film consumption for many consumers. In addition, the companies behind streamers are investing heavily into their own production projects. Amazon and Netflix have already begun this process. Disney and Apple - who have just launched their own streaming platforms - are sure to follow.

The dominance of franchises has been starkly clear over this decade with Marvel's Marvel Cinematic Universe (MCU) leading the way. While franchises have always performed well at the box office (see Star Wars, Jurrasic Park, Harry Potter), the MCU and Disney in particular have asserted their dominance in the 2010s. In 2019, out of the top 10 movies in terms of domestic gross box office, only the 10th ranked movie was an original idea with the rest being either sequels or part of a large franchise. 

Closely linked to franchise dominance is the shift toward tentpole productions has also happened. Movie studios are increasingly focusing on large-budget tentpole productions, especially established IP and cinematic unverses. The success of those projects allows them to bankroll other projects under the same IP or cinematic unverse umbrella. Smaller mid-budget films are thus being squeezed out of theaters and are ending up on streaming platforms or at smaller distributors.

### Key Objective

The main objective of this analysis is to identify the factors that positively affect a movie's ROI.

### Data Understanding and Analysis

#### Source of Data
The data used for this analysis was acquired from two sources:
* Movie budget and box office data from [The Numbers](https://www.the-numbers.com/)
* All other movie data such as release date, genres, and runtime from [IMDb](https://www.imdb.com/)

#### Data Cleaning

The datasets were faily clean and well-organized already so after some initial exploration and basic data cleaning I merged the table from The Numbers (TN) with one of the IMDb tables. The IMDb database file came with eight tables of which I only utilized 'movie_basics'. I merged the tables on the only common key available which was movie title.

``df_merged = pd.merge(df_tn, df_movie_basics, how='inner', on='title')``

#### Visualization of Data

A basic scatter of production budget against box office revenues confirms an intuition that says the more money you invest in producing a movie the more revenue that movie will generate.

![budget_revenue_scatter](https://github.com/user-attachments/assets/2c2c86be-47bc-4ed8-bbe0-9127d9624cf0)

Plotting that budget against ROI, however, reveals a relationship that is not very obvious.

![budget_roi_no_outliers](https://github.com/user-attachments/assets/e97fba5f-4116-4188-bcb5-a661ad1e6530)


#### Feature Engineering

Next I did some basic feature engineering to obtain the independent variables needed for my regression analysis. Since the independent variables are all categorical I converted to a quantitative form by using dummy variabless. Below is an example with the genre feature and the first five rows of the newly created genre columns:

``more_dummies = tn_imdb['genres'].str.get_dummies(sep=',')``

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Biography</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Fantasy</th>
      <th>...</th>
      <th>Music</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>News</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Sport</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

#### Results

With all of the categorical variables converted into quantitative form I was able to perform my OLS regression with ROI as the dependent variable. Below is a bar plot of the regression coefficients sorted in descending order.

![ROI_all_features_regression_coef](https://github.com/user-attachments/assets/6d66d791-bbcf-4837-9273-0ef5514292f7)

While the R-squared for this model is only 0.043, the results are not insignificant if statistically insignificant. The Horror and Mystery genres are especially interesting when coupling the regression findings with a plot of ROI distributions over genres. In the graph below I picked out the seven biggest movie genres and plotted the ROIs in those genres in a box plot. 

![roi_dist_genres_no_outliers](https://github.com/user-attachments/assets/b1cc08cf-930f-4c78-bf90-8e04405963b1)

The box plot provides support for the genre coefficient weights in the regression model. Horror movies, which are typically coupled with the Thriller tag as well, perform well when the success metric is ROI. A plot of production budget distributions over the same genres helps explain why. 

![budget_dist_genres_no_outliers](https://github.com/user-attachments/assets/c1bbe548-8aa6-47cd-ad9f-9bec7b72b9bd)

Horror films tend to be lower budget but still garner enough audience interest to do well in terms of ROI. 

In addition to just genre, movies that are released in the summer months or in the winter seem to have higher ROI. 

## Summary of Findings

The analysis in this project found that producing Horror movies tend to be excellent if the measure of success is ROI. Horror movies tend to be less costly to produce and still have big enough audiences to consume the films allowing for high ROI numbers. In addition to genre, releasing movies in the summer months (July, August) and winter seem to have the most positive effect on ROI. 
