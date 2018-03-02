# NCAA 2018 Mens March Madness

## 2018 NCAA March Madness Men's Basketball Predictions

### By Brice Walker

[View full project on nbviewer](http://nbviewer.jupyter.org/github/bricewalker/NCAA-2018-Mens-March-Madness/blob/master/Madness.ipynb)

## Outline

- [Introduction](#intro)
- [Feature extraction and engineering](#features)
- [Classification analysis](#classification)

<a id='intro'></a>

## Introduction

This is a classification project completed for the 2018 March Madness Kaggle Competition. In this project, I have extracted 18 season-based, and 28 tournament-based team-level characteristics from several datasets. I used datasets provided by kaggle as well as data scraped from sports-reference.com. I then engineered several advanced measures and extracted Elo ratings. I used these characteristics to predict probabilities for each matchup in the 2018 March Madness Schedule. My final model was a soft voting classifier that used KNeighbors, Random Forest, Extra Trees, Logistic Regression, Gradient Boosting, and LightGBM classifiers to predict probabilities for each matchup. I also ran XGBoost and Keras/Tensorflow neural network models.

<a id='features'></a>

## Feature Engineering and Extraction
This project attempts to predict outcomes of games based on the following team level characteristics for season games:

- A modified Elo rating (where new entrants are initialized at a score of 1500, and there is no reversion to the mean between seasons)
- Number of wins
- Avg points per game scored
- Avg points per game allowed
- Avg # of 3 pointers per game
- Avg turnovers per game
- Avg Assists per game
- Avg rebounds per game
- Avg steals per game
- Power 6 Conference
- Reg Season championships
- Strength of team's schedule
- Championship appearances
- Location of the game
- A simple rating system

And the following team level characteristics for tournament performance:<br>

> Note: If a team plays in more than one tourney in a year than these values are averaged over all tourneys they played that year.

- Tournament appearances
- Conference tournament championships
- Points scored for winning/losing team
- A measure of possession
- Offensive efficiency
- Defensive efficiency
- Net Rating (Offensive - Defensive efficiency
- Assist Ratio
- Turnover Ratio
- Shooting Percentage
- Effective Field Goal Percentage adjusting for the fact that 3pt shots are more valuable
- FTA Rating : How good a team is at drawing fouls.
- Percentage of team offensive rebounds
- Percentage of team defensive rebounds
- Percentage of team total rebounds

Finally, predictions using these features are weighted and stacked with predictions made through team trueskill ratings and a binary classification neural network trained in Keras/Tensorflow.

<a id='classification'></a>
## Classification Analysis
Predictive binary classification statistical models explored in this project include:

- Logistic Regression
- K-Nearest Neighbors
- Random Forests
- Extra Trees
- Support Vector Machines
- Gradient Boosting
- TensorFlow/Keras Neural Networks