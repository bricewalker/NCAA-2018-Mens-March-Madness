# Common imports
import pandas as pd
import numpy as np
import scipy as sp
import collections
import os
import sys
import math
from math import pi
from math import sqrt
import csv
import urllib
import pickle
import random
import statsmodels.api as sm
from patsy import dmatrices

# Math and descriptive stats
from math import sqrt
from scipy import stats
from scipy.stats import norm, skew
from scipy.stats.stats import pearsonr
from scipy.special import boxcox1p, inv_boxcox1p

# Sci-kit Learn modules for machine learning
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, log_loss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, VotingClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding
from sklearn.svm import SVR, LinearSVC, SVC, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA, KernelPCA

# Boosting libraries
import lightgbm as lgb
import xgboost

# Deep Learning modules
from keras.layers import Input, Dense, Dropout, Flatten, Embedding, merge, Activation
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.utils import np_utils, to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf

from trueskill import TrueSkill, Rating, rate_1vs1

K = 20.
HOME_ADVANTAGE = 100.

# Kaggle provided data
reg_season_compact_pd = pd.read_csv('Data/KaggleData/RegularSeasonCompactResults.csv')
seasons_pd = pd.read_csv('Data/KaggleData/Seasons.csv')
teams_pd = pd.read_csv('Data/KaggleData/Teams.csv')
tourney_compact_pd = pd.read_csv('Data/KaggleData/NCAATourneyCompactResults.csv')
tourney_detailed_pd = pd.read_csv('Data/KaggleData/NCAATourneyDetailedResults.csv')
conference_pd = pd.read_csv('Data/KaggleData/Conference.csv')
tourney_results_pd = pd.read_csv('Data/KaggleData/TourneyResults.csv')
sample_sub_pd = pd.read_csv('Data/KaggleData/sample_submission.csv')
tourney_seeds_pd = pd.read_csv('Data/KaggleData/NCAATourneySeeds.csv')
team_conferences_pd = pd.read_csv('Data/KaggleData/TeamConferences.csv')
sample_sub_pd = pd.read_csv('Data/KaggleData/SampleSubmissionStage1.csv')
seeds_pd = pd.read_csv('Data/KaggleData/NCAATourneySeeds.csv')
# Data I created
elos_ratings_pd = pd.read_csv('Data/Ratings/season_elos.csv')
enriched_pd = pd.read_csv('Data/KaggleData/NCAATourneyDetailedResultsEnriched.csv')

# Prelim stage 2 data
#tourney_seeds_pd = pd.read_csv('Data/KaggleData/NCAATourneySeeds_SampleTourney2018.csv')
#seasons_pd = pd.read_csv('Data/KaggleData/Seasons_SampleTourney2018.csv')
#slots_pd = pd.read_csv('Data/KaggleData/NCAATourneySlots_SampleTourney2018.csv')
#sample_sub_pd = pd.read_csv('Data/KaggleData/SampleSubmissionStage2_SampleTourney2018.csv')
#reg_season_compact_pd = pd.read_csv('Data/KaggleData/RegularSeasonCompactResults_Prelim2018.csv')
#reg_season_detailed_pd = pd.read_csv('Data/KaggleData/RegularSeasonDetailedResults_Prelim2018.csv')

# Final stage 2 data
reg_season_compact_pd = pd.read_csv('Data/Stage2UpdatedDataFiles/RegularSeasonCompactResults.csv')
seasons_pd = pd.read_csv('Data/Stage2UpdatedDataFiles/Seasons.csv')
teams_pd = pd.read_csv('Data/Stage2UpdatedDataFiles/Teams.csv')
slots_pd = pd.read_csv('Data/Stage2UpdatedDataFiles/NCAATourneySlots.csv')
seeds_pd = pd.read_csv('Data/Stage2UpdatedDataFiles/NCAATourneySeeds.csv')
reg_season_detailed_pd = pd.read_csv('Data/KaggleData/RegularSeasonDetailedResults.csv')
tourney_seeds_pd = pd.read_csv('Data/Stage2UpdatedDataFiles/NCAATourneySeeds.csv')
sample_sub_pd = pd.read_csv('Data/Stage2UpdatedDataFiles/SampleSubmissionStage2.csv')
# Data I created
enriched_pd = pd.read_csv('Data/Stage2UpdatedDataFiles/NCAATourneyDetailedResultsEnriched2018.csv')

def createTourneyFeats(): 
    # Advanced tournament data
    df = pd.read_csv('Data/KaggleData/NCAATourneyDetailedResults.csv')
    # Points Winning/Losing Team
    df['WPts'] = df.apply(lambda row: 2*row.WFGM + row.WFGM3 + row.WFTM, axis=1)
    df['LPts'] = df.apply(lambda row: 2*row.LFGM + row.LFGM3 + row.LFTM, axis=1)
    # Calculate Winning/losing Team Possesion Feature
    wPos = df.apply(lambda row: 0.96*(row.WFGA + row.WTO + 0.44*row.WFTA - row.WOR), axis=1)
    df['WPos'] = df.apply(lambda row: 0.96*(row.WFGA + row.WTO + 0.44*row.WFTA - row.WOR), axis=1)
    lPos = df.apply(lambda row: 0.96*(row.LFGA + row.LTO + 0.44*row.LFTA - row.LOR), axis=1)
    df['LPos'] = lPos = df.apply(lambda row: 0.96*(row.LFGA + row.LTO + 0.44*row.LFTA - row.LOR), axis=1)
    df['Pos'] = (wPos+lPos)/2
    # Offensive efficiency (OffRtg) = 100 x (Points / Possessions)
    df['WOffRtg'] = df.apply(lambda row: 100 * (row.WPts / row.Pos), axis=1)
    df['LOffRtg'] = df.apply(lambda row: 100 * (row.LPts / row.Pos), axis=1)
    # Defensive efficiency (DefRtg) = 100 x (Opponent points / Opponent possessions)
    df['WDefRtg'] = df.LOffRtg
    df['LDefRtg'] = df.WOffRtg
    # Net Rating = Off.Rtg - Def.Rtg
    df['WNetRtg'] = df.apply(lambda row:(row.WOffRtg - row.WDefRtg), axis=1)
    df['LNetRtg'] = df.apply(lambda row:(row.LOffRtg - row.LDefRtg), axis=1)                       
    # Assist Ratio : Percentage of team possessions that end in assists
    df['WAstR'] = df.apply(lambda row: 100 * row.WAst / (row.WFGA + 0.44*row.WFTA + row.WAst + row.WTO), axis=1)
    df['LAstR'] = df.apply(lambda row: 100 * row.LAst / (row.LFGA + 0.44*row.LFTA + row.LAst + row.LTO), axis=1)
    # Turnover Ratio: Number of turnovers of a team per 100 possessions used.
    # (TO * 100) / (FGA + (FTA * 0.44) + AST + TO
    df['WTOR'] = df.apply(lambda row: 100 * row.LAst / (row.LFGA + 0.44*row.LFTA + row.LAst + row.LTO), axis=1)
    df['LTOR'] = df.apply(lambda row: 100 * row.LAst / (row.LFGA + 0.44*row.LFTA + row.LAst + row.LTO), axis=1)                  
    # The Shooting Percentage : Measure of Shooting Efficiency (FGA/FGA3, FTA)
    df['WTSP'] = df.apply(lambda row: 100 * row.WPts / (2 * (row.WFGA + 0.44 * row.WFTA)), axis=1)
    df['LTSP'] = df.apply(lambda row: 100 * row.LPts / (2 * (row.LFGA + 0.44 * row.LFTA)), axis=1)
    # eFG% : Effective Field Goal Percentage adjusting for the fact that 3pt shots are more valuable 
    df['WeFGP'] = df.apply(lambda row:(row.WFGM + 0.5 * row.WFGM3) / row.WFGA, axis=1)      
    df['LeFGP'] = df.apply(lambda row:(row.LFGM + 0.5 * row.LFGM3) / row.LFGA, axis=1)   
    # FTA Rate : How good a team is at drawing fouls.
    df['WFTAR'] = df.apply(lambda row: row.WFTA / row.WFGA, axis=1)
    df['LFTAR'] = df.apply(lambda row: row.LFTA / row.LFGA, axis=1)                       
    # OREB% : Percentage of team offensive rebounds
    df['WORP'] = df.apply(lambda row: row.WOR / (row.WOR + row.LDR), axis=1)
    df['LORP'] = df.apply(lambda row: row.LOR / (row.LOR + row.WDR), axis=1)
    # DREB% : Percentage of team defensive rebounds
    df['WDRP'] = df.apply(lambda row: row.WDR / (row.WDR + row.LOR), axis=1)
    df['LDRP'] = df.apply(lambda row: row.LDR / (row.LDR + row.WOR), axis=1)                                      
    # REB% : Percentage of team total rebounds
    df['WRP'] = df.apply(lambda row: (row.WDR + row.WOR) / (row.WDR + row.WOR + row.LDR + row.LOR), axis=1)
    df['LRP'] = df.apply(lambda row: (row.LDR + row.WOR) / (row.WDR + row.WOR + row.LDR + row.LOR), axis=1) 
    df['WPIE'] = df.apply(lambda row: (row.WDR + row.WOR) / (row.WDR + row.WOR + row.LDR + row.LOR), axis=1)
    wtmp = df.apply(lambda row: row.WPts + row.WFGM + row.WFTM - row.WFGA - row.WFTA + row.WDR + 0.5*row.WOR + row.WAst +row.WStl + 0.5*row.WBlk - row.WPF - row.WTO, axis=1)
    ltmp = df.apply(lambda row: row.LPts + row.LFGM + row.LFTM - row.LFGA - row.LFTA + row.LDR + 0.5*row.LOR + row.LAst +row.LStl + 0.5*row.LBlk - row.LPF - row.LTO, axis=1) 
    df['WPIE'] = wtmp/(wtmp + ltmp)
    df['LPIE'] = ltmp/(wtmp + ltmp)

    df.to_csv('Data/Stage2UpdatedDataFiles/NCAATourneyDetailedResultsEnriched2018.csv', index=False)

def createEloRating():
    # Creating custom Elo ratings. This takes a long time to run so beware!
    team_ids = set(reg_season_compact_pd.WTeamID).union(set(reg_season_compact_pd.LTeamID))

    elo_dict = dict(zip(list(team_ids), [1500] * len(team_ids)))

    reg_season_compact_pd['margin'] = reg_season_compact_pd.WScore - reg_season_compact_pd.LScore
    reg_season_compact_pd['w_elo'] = None
    reg_season_compact_pd['l_elo'] = None

    def elo_pred(elo1, elo2):
        return(1. / (10. ** (-(elo1 - elo2) / 400.) + 1.))

    def expected_margin(elo_diff):
        return((7.5 + 0.006 * elo_diff))

    def elo_update(w_elo, l_elo, margin):
        elo_diff = w_elo - l_elo
        pred = elo_pred(w_elo, l_elo)
        mult = ((margin + 3.) ** 0.8) / expected_margin(elo_diff)
        update = K * mult * (1 - pred)
        return(pred, update)

    assert np.all(reg_season_compact_pd.index.values == np.array(range(reg_season_compact_pd.shape[0]))), "Index is out of order."

    preds = []

    # Loop over all rows
    for i in range(reg_season_compact_pd.shape[0]):

        # Get key data from each row
        w = reg_season_compact_pd.at[i, 'WTeamID']
        l = reg_season_compact_pd.at[i, 'LTeamID']
        margin = reg_season_compact_pd.at[i, 'margin']
        wloc = reg_season_compact_pd.at[i, 'WLoc']

        # Home court advantage?
        w_ad, l_ad, = 0., 0.
        if wloc == "H":
            w_ad += HOME_ADVANTAGE
        elif wloc == "A":
            l_ad += HOME_ADVANTAGE

        # Get elo updates as a result of each game
        pred, update = elo_update(elo_dict[w] + w_ad,
                                  elo_dict[l] + l_ad, 
                                  margin)
        elo_dict[w] += update
        elo_dict[l] -= update
        preds.append(pred)

        # Store elos in new dataframe
        reg_season_compact_pd.loc[i, 'w_elo'] = elo_dict[w]
        reg_season_compact_pd.loc[i, 'l_elo'] = elo_dict[l]

def seed_to_int(seed):
# Convert seeds to integers
    s_int = int(seed[1:3])
    return s_int
seeds_pd['seed_int'] = seeds_pd.Seed.apply(seed_to_int)
seeds_pd.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label

teamList = teams_pd['TeamName'].tolist()
NCAAChampionsList = tourney_results_pd['NCAA Champion'].tolist()

def Power6Conf(team_id):
    team_pd = team_conferences_pd[(team_conferences_pd['Season'] == 2018) & (team_conferences_pd['TeamID'] == team_id)]
    if (len(team_pd) == 0):
        return 0
    confName = team_pd.iloc[0]['ConfAbbrev']
    return int(confName == 'sec' or confName == 'acc'or confName == 'big_ten' or confName == 'big_twelve' or confName == 'big_east' or confName == 'pac_twelve')

def createTeamName(team_id):
    return teams_pd[teams_pd['TeamID'] == team_id].values[0][1]

def findNumChampionships(team_id):
    name = createTeamName(team_id)
    return NCAAChampionsList.count(name)

def handleCases(arr):
    indices = []
    listLen = len(arr)
    for i in range(listLen):
        if (arr[i] == 'St' or arr[i] == 'FL'):
            indices.append(i)
    for p in indices:
        arr[p-1] = arr[p-1] + ' ' + arr[p]
    for i in range(len(indices)): 
        arr.remove(arr[indices[i] - i])
    return arr

def checkConferenceChamp(team_id, year):
    year_conf_pd = conference_pd[conference_pd['Year'] == year]
    champs = year_conf_pd['Regular Season Champ'].tolist()
# In case of a tie
    champs_separated = [words for segments in champs for words in segments.split()]
    name = createTeamName(team_id)
    champs_separated = handleCases(champs_separated)
    if (name in champs_separated):
        return 1
    else:
        return 0

def checkConferenceTourneyChamp(team_id, year):
    year_conf_pd = conference_pd[conference_pd['Year'] == year]
    champs = year_conf_pd['Tournament Champ'].tolist()
    name = createTeamName(team_id)
    if (name in champs):
        return 1
    else:
        return 0

def getTourneyAppearances(team_id):
    return len(tourney_seeds_pd[tourney_seeds_pd['TeamID'] == team_id].index)

# Fixing names in csv's with differing formats
def handleDifferentCSV(df):
    df['School'] = df['School'].replace('(State)', 'St', regex=True) 
    df['School'] = df['School'].replace('Albany (NY)', 'Albany NY') 
    df['School'] = df['School'].replace('Boston University', 'Boston Univ')
    df['School'] = df['School'].replace('Central Michigan', 'C Michigan')
    df['School'] = df['School'].replace('(Eastern)', 'E', regex=True)
    df['School'] = df['School'].replace('Louisiana St', 'LSU')
    df['School'] = df['School'].replace('North Carolina St', 'NC State')
    df['School'] = df['School'].replace('Southern California', 'USC')
    df['School'] = df['School'].replace('University of California', 'California', regex=True) 
    df['School'] = df['School'].replace('American', 'American Univ')
    df['School'] = df['School'].replace('Arkansas-Little Rock', 'Ark Little Rock')
    df['School'] = df['School'].replace('Arkansas-Pine Bluff', 'Ark Pine Bluff')
    df['School'] = df['School'].replace('Bowling Green St', 'Bowling Green')
    df['School'] = df['School'].replace('Brigham Young', 'BYU')
    df['School'] = df['School'].replace('Cal Poly', 'Cal Poly SLO')
    df['School'] = df['School'].replace('Centenary (LA)', 'Centenary')
    df['School'] = df['School'].replace('Central Connecticut St', 'Central Conn')
    df['School'] = df['School'].replace('Charleston Southern', 'Charleston So')
    df['School'] = df['School'].replace('Coastal Carolina', 'Coastal Car')
    df['School'] = df['School'].replace('College of Charleston', 'Col Charleston')
    df['School'] = df['School'].replace('Cal St Fullerton', 'CS Fullerton')
    df['School'] = df['School'].replace('Cal St Sacramento', 'CS Sacramento')
    df['School'] = df['School'].replace('Cal St Bakersfield', 'CS Bakersfield')
    df['School'] = df['School'].replace('Cal St Northridge', 'CS Northridge')
    df['School'] = df['School'].replace('East Tennessee St', 'ETSU')
    df['School'] = df['School'].replace('Detroit Mercy', 'Detroit')
    df['School'] = df['School'].replace('Fairleigh Dickinson', 'F Dickinson')
    df['School'] = df['School'].replace('Florida Atlantic', 'FL Atlantic')
    df['School'] = df['School'].replace('Florida Gulf Coast', 'FL Gulf Coast')
    df['School'] = df['School'].replace('Florida International', 'Florida Intl')
    df['School'] = df['School'].replace('George Washington', 'G Washington')
    df['School'] = df['School'].replace('Georgia Southern', 'Ga Southern')
    df['School'] = df['School'].replace('Gardner-Webb', 'Gardner Webb')
    df['School'] = df['School'].replace('Illinois-Chicago', 'IL Chicago')
    df['School'] = df['School'].replace('Kent St', 'Kent')
    df['School'] = df['School'].replace('Long Island University', 'Long Island')
    df['School'] = df['School'].replace('Loyola Marymount', 'Loy Marymount')
    df['School'] = df['School'].replace('Loyola (MD)', 'Loyola MD')
    df['School'] = df['School'].replace('Loyola (IL)', 'Loyola-Chicago')
    df['School'] = df['School'].replace('Massachusetts', 'MA Lowell')
    df['School'] = df['School'].replace('Maryland-Eastern Shore', 'MD E Shore')
    df['School'] = df['School'].replace('Miami (FL)', 'Miami FL')
    df['School'] = df['School'].replace('Miami (OH)', 'Miami OH')
    df['School'] = df['School'].replace('Missouri-Kansas City', 'Missouri KC')
    df['School'] = df['School'].replace('Monmouth', 'Monmouth NJ')
    df['School'] = df['School'].replace('Mississippi Valley St', 'MS Valley St')
    df['School'] = df['School'].replace('Montana St', 'MTSU')
    df['School'] = df['School'].replace('Northern Colorado', 'N Colorado')
    df['School'] = df['School'].replace('North Dakota St', 'N Dakota St')
    df['School'] = df['School'].replace('Northern Illinois', 'N Illinois')
    df['School'] = df['School'].replace('Northern Kentucky', 'N Kentucky')
    df['School'] = df['School'].replace('North Carolina A&T', 'NC A&T')
    df['School'] = df['School'].replace('North Carolina Central', 'NC Central')
    df['School'] = df['School'].replace('Pennsylvania', 'Penn')
    df['School'] = df['School'].replace('South Carolina St', 'S Carolina St')
    df['School'] = df['School'].replace('Southern Illinois', 'S Illinois')
    df['School'] = df['School'].replace('UC-Santa Barbara', 'Santa Barbara')
    df['School'] = df['School'].replace('Southeastern Louisiana', 'SE Louisiana')
    df['School'] = df['School'].replace('Southeast Missouri St', 'SE Missouri St')
    df['School'] = df['School'].replace('Stephen F. Austin', 'SF Austin')
    df['School'] = df['School'].replace('Southern Methodist', 'SMU')
    df['School'] = df['School'].replace('Southern Mississippi', 'Southern Miss')
    df['School'] = df['School'].replace('Southern', 'Southern Univ')
    df['School'] = df['School'].replace('St. Bonaventure', 'St Bonaventure')
    df['School'] = df['School'].replace('St. Francis (NY)', 'St Francis NY')
    df['School'] = df['School'].replace('Saint Francis (PA)', 'St Francis PA')
    df['School'] = df['School'].replace('St. John\'s (NY)', 'St John\'s')
    df['School'] = df['School'].replace('Saint Joseph\'s', 'St Joseph\'s PA')
    df['School'] = df['School'].replace('Saint Louis', 'St Louis')
    df['School'] = df['School'].replace('Saint Mary\'s (CA)', 'St Mary\'s CA')
    df['School'] = df['School'].replace('Mount Saint Mary\'s', 'Mt St Mary\'s')
    df['School'] = df['School'].replace('Saint Peter\'s', 'St Peter\'s')
    df['School'] = df['School'].replace('Texas A&M-Corpus Christian', 'TAM C. Christian')
    df['School'] = df['School'].replace('Texas Christian', 'TCU')
    df['School'] = df['School'].replace('Tennessee-Martin', 'TN Martin')
    df['School'] = df['School'].replace('Texas-Rio Grande Valley', 'UTRGV')
    df['School'] = df['School'].replace('Texas Southern', 'TX Southern')
    df['School'] = df['School'].replace('Alabama-Birmingham', 'UAB')
    df['School'] = df['School'].replace('UC-Davis', 'UC Davis')
    df['School'] = df['School'].replace('UC-Irvine', 'UC Irvine')
    df['School'] = df['School'].replace('UC-Riverside', 'UC Riverside')
    df['School'] = df['School'].replace('Central Florida', 'UCF')
    df['School'] = df['School'].replace('Louisiana-Lafayette', 'ULL')
    df['School'] = df['School'].replace('Louisiana-Monroe', 'ULM')
    df['School'] = df['School'].replace('Maryland-Baltimore County', 'UMBC')
    df['School'] = df['School'].replace('North Carolina-Asheville', 'UNC Asheville')
    df['School'] = df['School'].replace('North Carolina-Greensboro', 'UNC Greensboro')
    df['School'] = df['School'].replace('North Carolina-Wilmington', 'UNC Wilmington')
    df['School'] = df['School'].replace('Nevada-Las Vegas', 'UNLV')
    df['School'] = df['School'].replace('Texas-Arlington', 'UT Arlington')
    df['School'] = df['School'].replace('Texas-San Antonio', 'UT San Antonio')
    df['School'] = df['School'].replace('Texas-El Paso', 'UTEP')
    df['School'] = df['School'].replace('Virginia Commonwealth', 'VA Commonwealth')
    df['School'] = df['School'].replace('Western Carolina', 'W Carolina')
    df['School'] = df['School'].replace('Western Illinois', 'W Illinois')
    df['School'] = df['School'].replace('Western Kentucky', 'WKU')
    df['School'] = df['School'].replace('Western Michigan', 'W Michigan')
    df['School'] = df['School'].replace('Abilene Christian', 'Abilene Chr')
    df['School'] = df['School'].replace('Montana State', 'Montana St')
    df['School'] = df['School'].replace('Central Arkansas', 'Cent Arkansas')
    df['School'] = df['School'].replace('Houston Baptist', 'Houston Bap')
    df['School'] = df['School'].replace('South Dakota St', 'S Dakota St')
    df['School'] = df['School'].replace('Maryland-Eastern Shore', 'MD E Shore')
    return df

def createHomeStat(row):
    if (row == 'H'):
        home = 1
    if (row == 'A'):
        home = -1
    if (row == 'N'):
        home = 0
    return home

def createTrueskillRating():
    ts = TrueSkill(draw_probability=0.01) # 0.01 is arbitary small number
    beta = 25 / 6  # default value

    def win_probability(p1, p2):
        delta_mu = p1.mu - p2.mu
        sum_sigma = p1.sigma * p1.sigma + p2.sigma * p2.sigma
        denom = np.sqrt(2 * (beta * beta) + sum_sigma)
        return ts.cdf(delta_mu / denom)

    submit = sample_sub_pd
    submit[['Season', 'Team1', 'Team2']] = submit.apply(lambda r:pd.Series([int(t) for t in r.ID.split('_')]), axis=1)

    df_tour = reg_season_compact_pd
    teamIds = np.unique(np.concatenate([df_tour.WTeamID.values, df_tour.LTeamID.values]))
    ratings = { tid:ts.Rating() for tid in teamIds }

    def feed_season_results(season):
        print("season = {}".format(season))
        df1 = df_tour[df_tour.Season == season]
        for r in df1.itertuples():
            ratings[r.WTeamID], ratings[r.LTeamID] = rate_1vs1(ratings[r.WTeamID], ratings[r.LTeamID])

    def update_pred(season):
        beta = np.std([r.mu for r in ratings.values()]) 
        print("beta = {}".format(beta))
        submit.loc[submit.Season==season, 'Pred'] = submit[submit.Season==season].apply(lambda r:win_probability(ratings[r.Team1], ratings[r.Team2]), axis=1)

    for season in sorted(df_tour.Season.unique()[:-1]): # exclude last 4 years [:-4]/ last 1 year [:-1]
        feed_season_results(season)

#    update_pred(2014)
#    feed_season_results(2014)
#    update_pred(2015)
#    feed_season_results(2015)
#    update_pred(2016)
#    feed_season_results(2016)
#    update_pred(2017)
    feed_season_results(2017)
    update_pred(2018)

    submit.drop(['Season', 'Team1', 'Team2'], axis=1, inplace=True)
    submit.to_csv('Data/Predictions/trueskill_results2018.csv', index=None)

def getSeasonTourneyData(team_id, year):
    year_data_pd = reg_season_compact_pd[reg_season_compact_pd['Season'] == year]
# Elo   
    year_pd = year_data_pd.copy()
    year_pd = year_pd.loc[(year_pd.WTeamID == team_id) | (year_pd.LTeamID == team_id), :]
    year_pd.sort_values(['Season', 'DayNum'], inplace=True)
    year_pd.drop_duplicates(['Season'], keep='last', inplace=True)
    w_mask = year_pd.WTeamID == team_id
    l_mask = year_pd.LTeamID == team_id
    year_pd['season_elo'] = None
    year_pd.loc[w_mask, 'season_elo'] = year_pd.loc[w_mask, 'w_elo']
    year_pd.loc[l_mask, 'season_elo'] = year_pd.loc[l_mask, 'l_elo']
    elo = year_pd.season_elo
    elo = elo.values.mean()
# Points per game
    gamesWon = year_data_pd[year_data_pd.WTeamID == team_id] 
    totalPointsScored = gamesWon['WScore'].sum()
    gamesLost = year_data_pd[year_data_pd.LTeamID == team_id] 
    totalGames = gamesWon.append(gamesLost)
    numGames = len(totalGames.index)
    totalPointsScored += gamesLost['LScore'].sum()
# Number of points allowed
    totalPointsAllowed = gamesWon['LScore'].sum()
    totalPointsAllowed += gamesLost['WScore'].sum()
# Scraped data    
    stats_SOS_pd = pd.read_csv('Data/RegSeasonStats/MMStats_'+str(year)+'.csv')
    stats_SOS_pd = handleDifferentCSV(stats_SOS_pd)
    ratings_pd = pd.read_csv('Data/RatingStats/RatingStats_'+str(year)+'.csv')
    ratings_pd = handleDifferentCSV(ratings_pd)
    
    name = createTeamName(team_id)
    team = stats_SOS_pd[stats_SOS_pd['School'] == name]
    team_rating = ratings_pd[ratings_pd['School'] == name]
    if (len(team.index) == 0 or len(team_rating.index) == 0):
        total3sMade = 0
        totalTurnovers = 0
        totalAssists = 0
        sos = 0
        totalRebounds = 0
        srs = 0
        totalSteals = 0
    else:
        total3sMade = team['X3P'].values[0]
        totalTurnovers = team['TOV'].values[0]
        if (math.isnan(totalTurnovers)):
            totalTurnovers = 0
        totalAssists = team['AST'].values[0]
        if (math.isnan(totalAssists)):
            totalAssists = 0
        sos = team['SOS'].values[0]
        srs = team['SRS'].values[0]
        totalRebounds = team['TRB'].values[0]
        if (math.isnan(totalRebounds)):
            totalRebounds = 0
        totalSteals = team['STL'].values[0]
        if (math.isnan(totalSteals)):
            totalSteals = 0
    
# Finding tourney seed
    tourneyYear = tourney_seeds_pd[tourney_seeds_pd['Season'] == year]
    seed = tourneyYear[tourneyYear['TeamID'] == team_id]
    if (len(seed.index) != 0):
        seed = seed.values[0][1]
        tournamentSeed = int(seed[1:3])
    else:
        tournamentSeed = 25

# Number of wins and losses
    numWins = len(gamesWon.index)
# Preventing division by 0
    if numGames == 0:
        avgPointsScored = 0
        avgPointsAllowed = 0
        avg3sMade = 0
        avgTurnovers = 0
        avgAssists = 0
        avgRebounds = 0
        avgSteals = 0
    else:
        avgPointsScored = totalPointsScored/numGames
        avgPointsAllowed = totalPointsAllowed/numGames
        avg3sMade = total3sMade/numGames
        avgTurnovers = totalTurnovers/numGames
        avgAssists = totalAssists/numGames
        avgRebounds = totalRebounds/numGames
        avgSteals = totalSteals/numGames
        
# Tourney data   
    enriched_df = enriched_pd[enriched_pd['Season'] == year]
    enriched_df = enriched_df.loc[(enriched_df.WTeamID == team_id) | (enriched_df.LTeamID == team_id), :]
    w_mask = enriched_df.WTeamID == team_id
    l_mask = enriched_df.LTeamID == team_id
    enriched_df['Score'] = 0
    enriched_df['FGM'] = 0
    enriched_df['FGA'] = 0
    enriched_df['FGM3'] = 0
    enriched_df['FGA3'] = 0
    enriched_df['FTM'] = 0
    enriched_df['FTA'] = 0
    enriched_df['OR'] = 0
    enriched_df['DR'] = 0
    enriched_df['Ast'] = 0
    enriched_df['TO'] = 0
    enriched_df['Stl'] = 0
    enriched_df['Blk'] = 0
    enriched_df['PF'] = 0
    enriched_df['Pts'] = 0
    enriched_df['Pos'] = 0
    enriched_df['OffRtg'] = 0
    enriched_df['DefRtg'] = 0
    enriched_df['NetRtg'] = 0
    enriched_df['AstR'] = 0
    enriched_df['TOR'] = 0
    enriched_df['TSP'] = 0
    enriched_df['eFGP'] = 0
    enriched_df['FTAR'] = 0
    enriched_df['ORP'] = 0
    enriched_df['DRP'] = 0
    enriched_df['RP'] = 0
    enriched_df['PIE'] = 0
    enriched_df.loc[w_mask, 'Score'] = enriched_df.loc[w_mask, 'WScore']
    enriched_df.loc[l_mask, 'Score'] = enriched_df.loc[l_mask, 'LScore']
    Score = enriched_df.Score.values.mean()
    enriched_df.loc[w_mask, 'FGM'] = enriched_df.loc[w_mask, 'WFGM']
    enriched_df.loc[l_mask, 'FGM'] = enriched_df.loc[l_mask, 'LFGM']
    FGM = enriched_df.FGM.values.mean()
    enriched_df.loc[w_mask, 'FGA'] = enriched_df.loc[w_mask, 'WFGA']
    enriched_df.loc[l_mask, 'FGA'] = enriched_df.loc[l_mask, 'LFGA']
    FGA = enriched_df.FGA.values.mean()
    enriched_df.loc[w_mask, 'FGM3'] = enriched_df.loc[w_mask, 'WFGM3']
    enriched_df.loc[l_mask, 'FGM3'] = enriched_df.loc[l_mask, 'LFGM3']
    FGM3 = enriched_df.FGM3.values.mean()
    enriched_df.loc[w_mask, 'FGA3'] = enriched_df.loc[w_mask, 'WFGA3']
    enriched_df.loc[l_mask, 'FGA3'] = enriched_df.loc[l_mask, 'LFGA3']
    FGA3 = enriched_df.FGA3.values.mean()
    enriched_df.loc[w_mask, 'FTM'] = enriched_df.loc[w_mask, 'WFTM']
    enriched_df.loc[l_mask, 'FTM'] = enriched_df.loc[l_mask, 'LFTM']
    FTM = enriched_df.FTM.values.mean()
    enriched_df.loc[w_mask, 'FTA'] = enriched_df.loc[w_mask, 'WFTA']
    enriched_df.loc[l_mask, 'FTA'] = enriched_df.loc[l_mask, 'LFTA']
    FTA = enriched_df.FTA.values.mean()
    enriched_df.loc[w_mask, 'OR'] = enriched_df.loc[w_mask, 'WOR']
    enriched_df.loc[l_mask, 'OR'] = enriched_df.loc[l_mask, 'LOR']
    OR = enriched_df.OR.values.mean()
    enriched_df.loc[w_mask, 'DR'] = enriched_df.loc[w_mask, 'WDR']
    enriched_df.loc[l_mask, 'DR'] = enriched_df.loc[l_mask, 'LDR']
    DR = enriched_df.DR.values.mean()
    enriched_df.loc[w_mask, 'Ast'] = enriched_df.loc[w_mask, 'WAst']
    enriched_df.loc[l_mask, 'Ast'] = enriched_df.loc[l_mask, 'LAst']
    Ast = enriched_df.Ast.values.mean()
    enriched_df.loc[w_mask, 'TO'] = enriched_df.loc[w_mask, 'WTO']
    enriched_df.loc[l_mask, 'TO'] = enriched_df.loc[l_mask, 'LTO']
    TO = enriched_df.TO.values.mean()
    enriched_df.loc[w_mask, 'Stl'] = enriched_df.loc[w_mask, 'WStl']
    enriched_df.loc[l_mask, 'Stl'] = enriched_df.loc[l_mask, 'LStl']
    Stl = enriched_df.Stl.values.mean()
    enriched_df.loc[w_mask, 'Blk'] = enriched_df.loc[w_mask, 'WBlk']
    enriched_df.loc[l_mask, 'Blk'] = enriched_df.loc[l_mask, 'LBlk']
    Blk = enriched_df.Blk.values.mean()
    enriched_df.loc[w_mask, 'PF'] = enriched_df.loc[w_mask, 'WPF']
    enriched_df.loc[l_mask, 'PF'] = enriched_df.loc[l_mask, 'LPF']
    PF = enriched_df.PF.values.mean()
    enriched_df.loc[w_mask, 'Pts'] = enriched_df.loc[w_mask, 'WPts']
    enriched_df.loc[l_mask, 'Pts'] = enriched_df.loc[l_mask, 'LPts']
    Pts = enriched_df.Pts.values.mean()
    enriched_df.loc[w_mask, 'Pos'] = enriched_df.loc[w_mask, 'WPos']
    enriched_df.loc[l_mask, 'Pos'] = enriched_df.loc[l_mask, 'LPos']
    Pos = enriched_df.Pos.values.mean()
    enriched_df.loc[w_mask, 'OffRtg'] = enriched_df.loc[w_mask, 'WOffRtg']
    enriched_df.loc[l_mask, 'OffRtg'] = enriched_df.loc[l_mask, 'LOffRtg']
    OffRtg = enriched_df.OffRtg.values.mean()
    enriched_df.loc[w_mask, 'DefRtg'] = enriched_df.loc[w_mask, 'WDefRtg']
    enriched_df.loc[l_mask, 'DefRtg'] = enriched_df.loc[l_mask, 'LDefRtg']
    DefRtg = enriched_df.DefRtg.values.mean()
    enriched_df.loc[w_mask, 'NetRtg'] = enriched_df.loc[w_mask, 'WNetRtg']
    enriched_df.loc[l_mask, 'NetRtg'] = enriched_df.loc[l_mask, 'LNetRtg']
    NetRtg = enriched_df.NetRtg.values.mean()
    enriched_df.loc[w_mask, 'AstR'] = enriched_df.loc[w_mask, 'WAstR']
    enriched_df.loc[l_mask, 'AstR'] = enriched_df.loc[l_mask, 'LAstR']
    AstR = enriched_df.AstR.values.mean()
    enriched_df.loc[w_mask, 'TOR'] = enriched_df.loc[w_mask, 'WTOR']
    enriched_df.loc[l_mask, 'TOR'] = enriched_df.loc[l_mask, 'LTOR']
    TOR = enriched_df.TOR.values.mean()
    enriched_df.loc[w_mask, 'TSP'] = enriched_df.loc[w_mask, 'WTSP']
    enriched_df.loc[l_mask, 'TSP'] = enriched_df.loc[l_mask, 'LTSP']
    TSP = enriched_df.TSP.values.mean()
    enriched_df.loc[w_mask, 'eFGP'] = enriched_df.loc[w_mask, 'WeFGP']
    enriched_df.loc[l_mask, 'eFGP'] = enriched_df.loc[l_mask, 'LeFGP']
    eFGP = enriched_df.eFGP.values.mean()
    enriched_df.loc[w_mask, 'FTAR'] = enriched_df.loc[w_mask, 'WFTAR']
    enriched_df.loc[l_mask, 'FTAR'] = enriched_df.loc[l_mask, 'LFTAR']
    FTAR = enriched_df.FTAR.values.mean()
    enriched_df.loc[w_mask, 'ORP'] = enriched_df.loc[w_mask, 'WORP']
    enriched_df.loc[l_mask, 'ORP'] = enriched_df.loc[l_mask, 'LORP']
    ORP = enriched_df.ORP.values.mean()
    enriched_df.loc[w_mask, 'DRP'] = enriched_df.loc[w_mask, 'WDRP']
    enriched_df.loc[l_mask, 'DRP'] = enriched_df.loc[l_mask, 'LDRP']
    DRP = enriched_df.DRP.values.mean()
    enriched_df.loc[w_mask, 'RP'] = enriched_df.loc[w_mask, 'WRP']
    enriched_df.loc[l_mask, 'RP'] = enriched_df.loc[l_mask, 'LRP']
    RP = enriched_df.RP.values.mean()
    enriched_df.loc[w_mask, 'PIE'] = enriched_df.loc[w_mask, 'WPIE']
    enriched_df.loc[l_mask, 'PIE'] = enriched_df.loc[l_mask, 'LPIE']
    PIE = enriched_df.PIE.values.mean()
     
    return [numWins, avgPointsScored, avgPointsAllowed, Power6Conf(team_id), avg3sMade, avgAssists, avgTurnovers,
            checkConferenceChamp(team_id, year), checkConferenceTourneyChamp(team_id, year), tournamentSeed,
            sos, srs, avgRebounds, avgSteals, getTourneyAppearances(team_id), findNumChampionships(team_id), elo,
            FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, PF, Pts, Pos, OffRtg, DefRtg, NetRtg, Score,
            AstR, TOR, TSP, eFGP, FTAR, ORP, DRP, RP, PIE, numWins * 2, avgPointsScored * 2, avgPointsAllowed * 2, 
            Power6Conf(team_id) * 2, avg3sMade * 2, avgAssists * 2, avgTurnovers * 2,
            checkConferenceChamp(team_id, year) * 2, checkConferenceTourneyChamp(team_id, year) * 2, 
            tournamentSeed * 2, sos * 2, srs * 2, avgRebounds * 2, avgSteals * 2, getTourneyAppearances(team_id) * 2, 
            findNumChampionships(team_id) * 2, elo * 2, FGM * 2, FGA * 2, FGM3 * 2, FGA3 * 2, FTM * 2, FTA * 2,
            OR * 2, DR * 2, Ast * 2, TO * 2, Stl * 2, Blk * 2, PF * 2, Pts * 2, Pos * 2, OffRtg * 2, DefRtg * 2, 
            NetRtg * 2, Score * 2, AstR * 2, TOR * 2, TSP * 2, eFGP * 2, FTAR * 2, ORP * 2, DRP * 2, RP * 2, 
            PIE * 2, numWins * 3, avgPointsScored * 3, avgPointsAllowed * 3, Power6Conf(team_id) * 3, 
            avg3sMade * 3, avgAssists * 3, avgTurnovers * 3, checkConferenceChamp(team_id, year) * 3, 
            checkConferenceTourneyChamp(team_id, year) * 3, tournamentSeed * 3, sos * 3, srs * 3, avgRebounds * 3,
            avgSteals * 3, getTourneyAppearances(team_id) * 3, findNumChampionships(team_id) * 3, elo * 3, FGM * 3,
            FGA * 3, FGM3 * 3, FGA3 * 3, FTM * 3, FTA * 3, OR * 3, DR * 3, Ast * 3, TO * 3, Stl * 3, Blk * 3,
            PF * 3, Pts * 3, Pos * 3, OffRtg * 3, DefRtg * 3, NetRtg * 3, Score * 3, AstR * 3, TOR * 3, TSP * 3,
            eFGP * 3, FTAR * 3, ORP * 3, DRP * 3, RP * 3, PIE * 3]

def compareTwoTeams(id_1, id_2, year):
    team_1 = getSeasonTourneyData(id_1, year)
    team_2 = getSeasonTourneyData(id_2, year)
    diff = [a - b for a, b in zip(team_1, team_2)]
    return diff

def createStatDict(year):
    statDictionary = collections.defaultdict(list)
    for team in teamList:
        team_id = teams_pd[teams_pd['TeamName'] == team].values[0][0]
        team_vector = getSeasonTourneyData(team_id, year)
        statDictionary[team_id] = team_vector
    return statDictionary

def createTrainingSet(years, stage1Years, Stage2Year):
    createTourneyFeats()
    createEloRating()
    createTrueskillRating()
    totalNumGames = 0
    for year in years:
        season = reg_season_compact_pd[reg_season_compact_pd['Season'] == year]
        totalNumGames += len(season.index)
        tourney = tourney_compact_pd[tourney_compact_pd['Season'] == year]
        totalNumGames += len(tourney.index)
    numFeatures = len(getSeasonTourneyData(1181,2012))
    X_train = np.zeros(( totalNumGames, numFeatures + 1))
    y_train = np.zeros(( totalNumGames ))
    indexCounter = 0
    for year in years:
        team_vectors = createStatDict(year)
        season = reg_season_compact_pd[reg_season_compact_pd['Season'] == year]
        numGamesInSeason = len(season.index)
        tourney = tourney_compact_pd[tourney_compact_pd['Season'] == year]
        numGamesInSeason += len(tourney.index)
        xTrainSeason = np.zeros(( numGamesInSeason, numFeatures + 1))
        yTrainSeason = np.zeros(( numGamesInSeason ))
        counter = 0
        for index, row in season.iterrows():
            w_team = row['WTeamID']
            w_vector = team_vectors[w_team]
            l_team = row['LTeamID']
            l_vector = team_vectors[l_team]
            diff = [a - b for a, b in zip(w_vector, l_vector)]
            home = createHomeStat(row['WLoc'])
            if (counter % 2 == 0):
                diff.append(home) 
                xTrainSeason[counter] = diff
                yTrainSeason[counter] = 1
            else:
                diff.append(-home)
                xTrainSeason[counter] = [ -p for p in diff]
                yTrainSeason[counter] = 0
            counter += 1
        for index, row in tourney.iterrows():
            w_team = row['WTeamID']
            w_vector = team_vectors[w_team]
            l_team = row['LTeamID']
            l_vector = team_vectors[l_team]
            diff = [a - b for a, b in zip(w_vector, l_vector)]
            home = 0
            if (counter % 2 == 0):
                diff.append(home) 
                xTrainSeason[counter] = diff
                yTrainSeason[counter] = 1
            else:
                diff.append(-home)
                xTrainSeason[counter] = [ -p for p in diff]
                yTrainSeason[counter] = 0
            counter += 1
        X_train[indexCounter:numGamesInSeason+indexCounter] = xTrainSeason
        y_train[indexCounter:numGamesInSeason+indexCounter] = yTrainSeason
        indexCounter += numGamesInSeason
        print ('Finished year:', year)
        if (year in stage1Years):
            np.save('Data/PrecomputedMatrices/TeamVectors/' + str(year) + 'TeamVectors', team_vectors)
        if (year == stage2Year):
            np.save('Data/PrecomputedMatrices/Stage2/' + str(year) + 'TeamVectors', team_vectors)
    return X_train, y_train

def createAndSave(years, stage1Years, stage2Year):
    X_train, y_train = createTrainingSet(years, stage1Years, stage2Year)
    np.save('Data/PrecomputedMatrices/X_train', X_train)
    np.save('Data/PrecomputedMatrices/y_train', y_train)

years = range(1994,2019)
# Saves the team vectors for the following years
stage1Years = range(2014,2018)
stage2Year = 2018
if os.path.exists("Data/PrecomputedMatrices/X_train.npy") and os.path.exists("Data/PrecomputedMatrices/y_train.npy"):
    print ('There are already precomputed X_train, and y_train matricies.')
    os.remove("Data/PrecomputedMatrices/X_train.npy")
    os.remove("Data/PrecomputedMatrices/y_train.npy")
    createAndSave(years, stage1Years, stage2Year)
else:
    createAndSave(years, stage1Years, stage2Year)