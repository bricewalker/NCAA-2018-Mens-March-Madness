# NCAA 2018 Mens March Madness

## 2018 NCAA March Madness Men's Basketball Predictions

### By Brice Walker

[View full project on nbviewer](http://nbviewer.jupyter.org/github/bricewalker/NCAA-2018-Mens-March-Madness/blob/master/Madness.ipynb)

## Outline
- [Getting Started](#start)
- [Introduction](#intro)
- [Feature extraction and engineering](#features)
- [Classification analysis](#classification)

<a id='start'></a>

## Getting Started
I have provided a few handy scripts to allow quickly running this on your own. This project was built/tested on python 3.6 and it is recommended to use 3.6+. It is also recommended that you use Anaconda. This project requires Jupyter Notebook.

Simply run the following codes from a terminal in the repo directory:

```
pip install requirements.txt
python process_data.py
python train_model.py
```
> Note: You may need to [install XGBoost from source](https://github.com/dmlc/xgboost).<br>
> You may also need to run ```pip uninstall tensorflow``` to uninstall tensorflow and then run ```pip install tensorflow-gpu``` in order to take advantage of TensorFlow's GPU acceleration.

I have also provided a jupyter notebook that walks you through the iterative process.

<a id='intro'></a>

## Introduction
This is a classification project completed for the 2018 March Madness Kaggle Competition. In this project, I have extracted 18 season-based, and 28 tournament-based team-level characteristics from several datasets using data from 1994-2017. I used datasets provided by kaggle, as well as data scraped from sports-reference.com. I then engineered several advanced measures and extracted Elo ratings. I used these characteristics to predict probabilities for each matchup in the 2018 March Madness Schedule. I then created a well calibrated soft voting classifier that used KNeighbors, Random Forest, Extra Trees, Logistic Regression, Gradient Boosting, and LightGBM classifiers as well as a Keras/TensorFlow Neural Network to predict probabilities for each matchup. Finally, I developed predictions based on Microsoft's TrueSkill rating system and weighted them with the machine learning model predictions.

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
- Net Rating (Offensive - Defensive efficiency)
- Assist Ratio
- Turnover Ratio
- Shooting Percentage
- Effective Field Goal Percentage adjusting for the fact that 3pt shots are more valuable
- FTA Rating : How good a team is at drawing fouls.
- Percentage of team offensive rebounds
- Percentage of team defensive rebounds
- Percentage of team total rebounds

Finally, predictions using these features are weighted and stacked with predictions made through team trueskill ratings.

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
- Principal Component Analysis
- Ensembling/Stacking and Weighting Models