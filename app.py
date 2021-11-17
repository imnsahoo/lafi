from datetime import datetime
from flask import Flask,render_template
#from django.shortcuts import render
#from django.http import HttpResponse
import numpy as np 
from numpy import random
import pandas as pd 
#import matplotlib.pyplot as plt
#import matplotlib as mpl
#import seaborn as sns
#import proplot as pplt                        
from scipy import stats                      # Remove outliers
#import matplotlib.ticker as ticker           
from sklearn.preprocessing import MinMaxScaler
#import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import calendar
#import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
from sklearn.ensemble import  RandomForestRegressor
from sklearn import linear_model
from sklearn.metrics import mean_absolute_percentage_error
import warnings
#import missingno as msno
import pickle
import xgboost as xgb
from sklearn import model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error as MSE
#from django.shortcuts import render
from plotly.offline import plot
from plotly.graph_objs import Scatter
from flask import request
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash_application import create_dash_application
from flask import Blueprint
warnings.filterwarnings('ignore') 

app = Flask(__name__)

@app.route("/")
def hello_world():     
    return render_template("index.html")    

@app.route("/LFAI")
def LFAI():
    train = pd.read_csv('TRAIN_DATA_1_ACTIVITIES.csv')
    test = pd.read_csv('Test_data_Activity_Wise.csv')

    storeChoosen   = request.args['Stores']
    deptChoosen    = request.args['Departments']
    startDate      = pd.to_datetime(request.args['startDate'])
    endDate        = pd.to_datetime(request.args['endDate'])
    # FTE_var        = request.args['FTE']
    PTE_var        = request.args.get("PTE")
    Individual     = request.args['Individual']
    FTE_value       = request.args.get("FTE")
    multiRole     = request.args.getlist('Role')

    
    plot_div2 = '' 
    plot_div  =''
    plot_div1 =''
    futurePred = False;
    
    dateRangeText  = 'From '+request.args['startDate']+' To '+request.args['endDate']

    train['Date'] = pd.to_datetime(train.Date) 
    selected_dept = deptChoosen
    start_date    = pd.to_datetime(startDate)
    end_date      = pd.to_datetime(endDate)
    mask          = (train['Date'] >= start_date) & (train['Date'] <= end_date)
    test          = train.loc[mask]
    test          = test.loc[train['Department'] == selected_dept]
    Depts         = test.Department.values
    test          = test.reset_index(drop=True)
    
    
    if test.empty:
        futurePred    = True
        startDate       = request.args['startDate'].replace('2021', '2019')
        endDate         = request.args['endDate'].replace('2021', '2019')
        start_date      = pd.to_datetime(startDate)
        end_date        = pd.to_datetime(endDate)
        mask            = (train['Date'] > start_date) & (train['Date'] < end_date)
        test            = train.loc[mask]
        test            = test.loc[train['Department'] == selected_dept]
        Depts           = test.Department.values
        test            = test.reset_index(drop=True)
            
    actual_hour = pd.DataFrame()
    actual_hour['Department'] =test['Department']
    actual_hour['Working_Hour'] = test['Working_Hour']
    actual_hour['Cleaning'] = test['Cleaning']
    actual_hour['Cashier'] = test['Cashier']
    actual_hour['Picking'] = test['Picking']
    actual_hour['Reset'] = test['Reset']
    actual_hour['Back_Offc'] = test['Back_Offc']
    actual_hour['HR'] = test['HR']
    actual_hour['Security'] = test['Security']
    actual_hour['FTE'] = test['FTE']
    actual_hour['PTE'] = test['PTE']
    actual_hour['Date'] = test['Date']

    train['Promotion'] = train['Promotion'].map({'Yes': 1, 'No': 0}).astype(str)
    train['Holiday'] = train['Holiday'].map({'Yes': 1, 'No': 0}).astype(str)

    test['Promotion'] = test['Promotion'].map({'Yes': 1, 'No': 0}).astype(str)
    test['Holiday'] = test['Holiday'].map({'Yes': 1, 'No': 0}).astype(str)

    test[['Store', 'Holiday', 'Promotion']] = test[['Store', 'Holiday', 'Promotion']].astype(object)
    test['Date'] = pd.to_datetime(test.Date)

    train[['Store', 'Holiday', 'Promotion']] = train[['Store', 'Holiday', 'Promotion']].astype(object)
    train['Date'] = pd.to_datetime(train.Date)

    test=test[train.Order != 0.0]
    test=test[train.Working_Hour != 0.0]
    test=test[train.Sales != 0.0]

    train=train[train.Order != 0.0]
    train=train[train.Working_Hour != 0.0]
    train=train[train.Sales != 0.0]

    train['Day_of_week'] = train['Date'].dt.dayofweek
    train['Day'] = train['Date'].dt.month.apply(lambda x: calendar.month_abbr[x]) +'_'+train['Date'].dt.day.astype(str)

    train_missing= missing_values_table(train)
    train.fillna(0,inplace=True)

    test = test.drop('Store', 1)
    train = train.drop('Store', 1)
    train = train.drop('Day_of_week', 1)
    train = train.drop('Day', 1)

    # Data types
    train[['Holiday', 'Promotion']] = train[[ 'Holiday', 'Promotion']].astype(int)
    train['PTE'] = train['PTE'].astype(np.int64)
    train['Reset'] = train['Reset'].astype(np.int64)
    train['PTE'] = train['PTE'].astype(np.int64)
    test['PTE'] = test['PTE'].astype(np.int64)

    test[['Holiday', 'Promotion']] = test[[ 'Holiday', 'Promotion']].astype(int)


    categ_cols = test.dtypes[test.dtypes == np.object]                  # filtering by categorical variables
    categ_cols = categ_cols.index.tolist()                              # list of categorical fields
    test = pd.get_dummies(test, columns=categ_cols, drop_first=False) 

    categ_cols = train.dtypes[train.dtypes == np.object]                  # filtering by categorical variables
    categ_cols = categ_cols.index.tolist()                                # list of categorical fields
    train = pd.get_dummies(train, columns=categ_cols, drop_first=False)   # One hot encoding

    test['Year'] = test['Date'].dt.year
    test['Day_of_week'] = test['Date'].dt.dayofweek
    test['Weekday'] = test['Date'].dt.weekday
    test = test.drop(['Date'], axis = 1) 

    test = pd.get_dummies(test, columns=['Year'], drop_first=False, prefix='Year')
    test = pd.get_dummies(test, columns=['Day_of_week'], drop_first=False, prefix='Day_of_week')
    test = pd.get_dummies(test, columns=['Weekday'], drop_first=False, prefix='Weekday')

    train['Year'] = train['Date'].dt.year
    train['Day_of_week'] = train['Date'].dt.dayofweek
    train['Weekday'] = train['Date'].dt.weekday
    #train['Month'] = train['Date'].dt.month
    #train['Quarter'] = train['Date'].dt.quarter

    train = train.drop(['Date'], axis = 1) 

    train = pd.get_dummies(train, columns=['Year'], drop_first=False, prefix='Year')
    train = pd.get_dummies(train, columns=['Day_of_week'], drop_first=False, prefix='Day_of_week')
    train = pd.get_dummies(train, columns=['Weekday'], drop_first=False, prefix='Weekday')

    train_cols = train.columns
    test_cols = test.columns

    common_cols = train_cols.intersection(test_cols)
    train_not_test = train_cols.difference(test_cols)
    test[train_not_test] = 0
    test = test.reindex(sorted(test.columns), axis=1)
    train = train.reindex(sorted(train.columns), axis=1)


    train_y = pd.DataFrame()
    train_y ['Working_Hour'] = train['Working_Hour']
    train_y ['Cleaning'] = train['Cleaning']
    train_y ['Cashier'] = train['Cashier']
    train_y ['Picking'] = train['Picking']
    train_y ['Reset'] = train['Reset']
    train_y ['Back_Offc'] = train['Back_Offc']
    train_y ['HR'] = train['HR']
    train_y ['Security'] = train['Security']
    train_y ['FTE'] = train['FTE']
    train_y ['PTE'] = train['PTE']

    train_X = train.drop(["Working_Hour",'Cleaning','Cashier','Picking','Reset','Back_Offc','HR','Security','FTE','PTE'],axis=1)
    back_X = train_X
    x_test = test.drop(["Working_Hour",'Cleaning','Cashier','Picking','Reset','Back_Offc','HR','Security','FTE','PTE'],axis=1)

    xgbFTEmodel = xgb.XGBRegressor()
    xgbFTEmodel.fit(train_X,train_y['FTE'])
    xgb_pred = xgbFTEmodel.predict(x_test)

    finaXgb_Pred = pd.DataFrame()
    finaXgb_Pred["Dept"] = Depts
    finaXgb_Pred["Date"] = actual_hour['Date']
    finaXgb_Pred["FTE"] = xgb_pred
    finaXgb_Pred['FTE_ACT'] = actual_hour['FTE']
    
    xgbPTEmodel = xgb.XGBRegressor()
    xgbPTEmodel.fit(train_X,train_y['PTE'])
    xgbPTE_pred = xgbPTEmodel.predict(x_test)
    finaXgbPTE_Pred = pd.DataFrame()
    finaXgbPTE_Pred["Dept"] = Depts
    finaXgbPTE_Pred["Date"] = actual_hour['Date']
    finaXgbPTE_Pred["PTE"] = xgbPTE_pred
    finaXgbPTE_Pred['PTE_ACT'] = actual_hour['PTE']

    xgbSecuritymodel = xgb.XGBRegressor()
    xgbSecuritymodel.fit(train_X,train_y['Security'])
    xgbSecurity_pred = xgbSecuritymodel.predict(x_test)
    finaXgbSecurity_Pred = pd.DataFrame()
    finaXgbSecurity_Pred["Dept"] = Depts
    finaXgbSecurity_Pred["Date"] = actual_hour['Date']
    finaXgbSecurity_Pred["Security"] = xgbSecurity_pred
    finaXgbSecurity_Pred['Security_ACT'] = actual_hour['Security']
    
    xgbHRmodel = xgb.XGBRegressor()
    xgbHRmodel.fit(train_X,train_y['HR'])
    xgbHR_pred = xgbHRmodel.predict(x_test)
    finaXgbHR_Pred = pd.DataFrame()
    finaXgbHR_Pred["Dept"] = Depts
    finaXgb_Pred["Date"] = actual_hour['Date']
    finaXgbHR_Pred["HR"] = xgbHR_pred
    finaXgbHR_Pred['HR_ACT'] = actual_hour['HR']

    xgbBack_Offcmodel = xgb.XGBRegressor()
    xgbBack_Offcmodel.fit(train_X,train_y['Back_Offc'])
    xgbBackOffc_pred = xgbBack_Offcmodel.predict(x_test)
    finaXgbBackOffc_Pred = pd.DataFrame()
    finaXgbBackOffc_Pred["Dept"] = Depts
    finaXgbBackOffc_Pred["Date"] = actual_hour['Date']
    finaXgbBackOffc_Pred["Back_Offc"] = xgbBackOffc_pred
    finaXgbBackOffc_Pred['Back_Offc_ACT'] = actual_hour['Back_Offc']

    xgbResetmodel = xgb.XGBRegressor()
    xgbResetmodel.fit(train_X,train_y['Reset'])
    xgbReset_pred = xgbResetmodel.predict(x_test)
    finaXgbReset_Pred = pd.DataFrame()
    finaXgbReset_Pred["Dept"] = Depts
    finaXgbReset_Pred["Date"] = actual_hour['Date']
    finaXgbReset_Pred["Reset"] = xgbReset_pred
    finaXgbReset_Pred['Reset_ACT'] = actual_hour['Reset']

    xgbPickingmodel = xgb.XGBRegressor()
    xgbPickingmodel.fit(train_X,train_y['Picking'])
    xgbPicking_pred = xgbPickingmodel.predict(x_test)
    finaXgbPicking_Pred = pd.DataFrame()
    finaXgbPicking_Pred["Dept"] = Depts
    finaXgbPicking_Pred["Date"] = actual_hour['Date']
    finaXgbPicking_Pred["Picking"] = xgbPicking_pred
    finaXgbPicking_Pred['Picking_ACT'] = actual_hour['Picking']

    xgbCashiermodel = xgb.XGBRegressor()
    xgbCashiermodel.fit(train_X,train_y['Cashier'])
    xgbCashier_pred = xgbCashiermodel.predict(x_test)
    finaXgbCashier_Pred = pd.DataFrame()
    finaXgbCashier_Pred["Dept"] = Depts
    finaXgbCashier_Pred["Date"] = actual_hour['Date']
    finaXgbCashier_Pred["Cashier"] = xgbCashier_pred
    finaXgbCashier_Pred['Cashier_ACT'] = actual_hour['Cashier']

    xgbCleaningmodel = xgb.XGBRegressor()
    xgbCleaningmodel.fit(train_X,train_y['Cleaning'])
    xgbCleaning_pred = xgbCleaningmodel.predict(x_test)
    finaXgbCleaning_Pred = pd.DataFrame()
    finaXgbCleaning_Pred["Dept"] = Depts
    finaXgbCleaning_Pred["Date"] = actual_hour['Date']
    finaXgbCleaning_Pred["Cleaning"] = xgbCleaning_pred
    finaXgbCleaning_Pred['Cleaning_ACT'] = actual_hour['Cleaning']

    xgbWHmodel = xgb.XGBRegressor()
    xgbWHmodel.fit(train_X,train_y['Working_Hour'])
    xgbWH_pred = xgbWHmodel.predict(x_test)
    finaXgbWH_Pred = pd.DataFrame()
    finaXgbWH_Pred["Dept"] = Depts
    finaXgbWH_Pred["Date"] = actual_hour['Date']
    finaXgbWH_Pred["Working_Hour"] = xgbWH_pred
    finaXgbWH_Pred['Working_Hour_ACT'] = actual_hour['Working_Hour']

    # Prediction using Random Forest Algorithm

    model_rf = RandomForestRegressor(random_state=42)
    model_rf.fit(train_X, train_y['Working_Hour'])
    pred_rf = model_rf.predict(x_test)
    rf_Pred = pd.DataFrame()
    rf_Pred["Dept"] = Depts
    rf_Pred["Date"] = actual_hour['Date']
    rf_Pred["Working_Hour"] = pred_rf
    rf_Pred['Working_Hour_ACT'] = actual_hour['Working_Hour']

    modelCleaning_rf = RandomForestRegressor(random_state=42)
    modelCleaning_rf.fit(train_X, train_y['Cleaning'])
    predCleaning_rf = modelCleaning_rf.predict(x_test)
    rfCleaning_Pred = pd.DataFrame()
    rfCleaning_Pred["Dept"] = Depts
    rfCleaning_Pred["Date"] = actual_hour['Date']
    rfCleaning_Pred["Cleaning"] = predCleaning_rf
    rfCleaning_Pred['Cleaning_ACT'] = actual_hour['Cleaning']

    modelCashier_rf = RandomForestRegressor(random_state=42)
    modelCashier_rf.fit(train_X, train_y['Cashier'])
    predCashier_rf = modelCashier_rf.predict(x_test)
    rfCashier_Pred = pd.DataFrame()
    rfCashier_Pred["Dept"] = Depts
    rfCashier_Pred["Date"] = actual_hour['Date']
    rfCashier_Pred["Cashier"] = predCashier_rf
    rfCashier_Pred['Cashier_ACT'] = actual_hour['Cashier']

    modelBack_Offc_rf = RandomForestRegressor(random_state=42)
    modelBack_Offc_rf.fit(train_X, train_y['Back_Offc'])
    predBack_Offc_rf = modelBack_Offc_rf.predict(x_test)
    rfBack_Offc_Pred = pd.DataFrame()
    rfBack_Offc_Pred["Dept"] = Depts
    rfBack_Offc_Pred["Date"] = actual_hour['Date']
    rfBack_Offc_Pred["Back_Offc"] = predBack_Offc_rf
    rfBack_Offc_Pred['Back_Offc_ACT'] = actual_hour['Back_Offc']

    modelPicking_rf = RandomForestRegressor(random_state=42)
    modelPicking_rf.fit(train_X, train_y['Picking'])
    predPicking_rf = modelPicking_rf.predict(x_test)
    rfPicking_Pred = pd.DataFrame()
    rfPicking_Pred["Dept"] = Depts
    rfPicking_Pred["Date"] = actual_hour['Date']
    rfPicking_Pred["Picking"] = predPicking_rf
    rfPicking_Pred['Picking_ACT'] = actual_hour['Picking']

    modelReset_rf = RandomForestRegressor(random_state=42)
    modelReset_rf.fit(train_X, train_y['Reset'])
    predReset_rf = modelReset_rf.predict(x_test)
    rfReset_Pred = pd.DataFrame()
    rfReset_Pred["Dept"] = Depts
    rfReset_Pred["Date"] = actual_hour['Date']
    rfReset_Pred["Reset"] = predReset_rf
    rfReset_Pred['Reset_ACT'] = actual_hour['Reset']

    modelHR_rf = RandomForestRegressor(random_state=42)
    modelHR_rf.fit(train_X, train_y['HR'])
    predHR_rf = modelHR_rf.predict(x_test)
    rfHR_Pred = pd.DataFrame()
    rfHR_Pred["Dept"] = Depts
    rfHR_Pred["Date"] = actual_hour['Date']
    rfHR_Pred["HR"] = predHR_rf
    rfHR_Pred['HR_ACT'] = actual_hour['HR']

    modelSecurity_rf = RandomForestRegressor(random_state=42)
    modelSecurity_rf.fit(train_X, train_y['Security'])
    predmodelSecurity_rf_rf = modelSecurity_rf.predict(x_test)
    rfSecurity_Pred = pd.DataFrame()
    rfSecurity_Pred["Dept"] = Depts
    rfSecurity_Pred["Date"] = actual_hour['Date']
    rfSecurity_Pred["Security"] = predmodelSecurity_rf_rf
    rfSecurity_Pred['Security_ACT'] = actual_hour['Security']

    modelPTE_rf = RandomForestRegressor(random_state=42)
    modelPTE_rf.fit(train_X, train_y['PTE'])
    predPTE_rf = modelPTE_rf.predict(x_test)
    rfPTE_Pred = pd.DataFrame()
    rfPTE_Pred["Dept"] = Depts
    rfPTE_Pred["Date"] = actual_hour['Date']
    rfPTE_Pred["PTE"] = predPTE_rf
    rfPTE_Pred['PTE_ACT'] = actual_hour['PTE']

    modelFTE_rf = RandomForestRegressor(random_state=42)
    modelFTE_rf.fit(train_X, train_y['FTE'])
    predFTE_rf = modelFTE_rf.predict(x_test)
    rfFTE_Pred = pd.DataFrame()
    rfFTE_Pred["Dept"] = Depts
    rfFTE_Pred["Date"] = actual_hour['Date']
    rfFTE_Pred["FTE"] = predFTE_rf
    rfFTE_Pred['FTE_ACT'] = actual_hour['FTE']

    from sklearn.tree import DecisionTreeRegressor
    model_dt = DecisionTreeRegressor()
    model_dt.fit(train_X, train_y['Working_Hour'])
    dt_pred = model_dt.predict(x_test)
    dtree_Pred = pd.DataFrame()
    dtree_Pred["Dept"] = Depts
    dtree_Pred["Date"] = actual_hour['Date']
    dtree_Pred["Working_Hour"] = dt_pred
    dtree_Pred['Working_Hour_ACT'] = actual_hour['Working_Hour']

    modelCleaning_dt = DecisionTreeRegressor()
    modelCleaning_dt.fit(train_X, train_y['Cleaning'])
    dtCleaning_pred = modelCleaning_dt.predict(x_test)
    dtreeCleaning_Pred = pd.DataFrame()
    dtreeCleaning_Pred["Dept"] = Depts
    dtreeCleaning_Pred["Date"] = actual_hour['Date']
    dtreeCleaning_Pred["Cleaning"] = dtCleaning_pred
    dtreeCleaning_Pred['Cleaning_ACT'] = actual_hour['Cleaning']

    modelCashier_dt = DecisionTreeRegressor()
    modelCashier_dt.fit(train_X, train_y['Cashier'])
    dtCashier_pred = modelCashier_dt.predict(x_test)
    dtreeCashier_Pred = pd.DataFrame()
    dtreeCashier_Pred["Dept"] = Depts
    dtreeCashier_Pred["Date"] = actual_hour['Date']
    dtreeCashier_Pred["Cashier"] = dtCashier_pred
    dtreeCashier_Pred['Cashier_ACT'] = actual_hour['Cashier']


    modelPicking_dt = DecisionTreeRegressor()
    modelPicking_dt.fit(train_X, train_y['Picking'])
    dtPicking_pred = modelPicking_dt.predict(x_test)
    dtreePicking_Pred = pd.DataFrame()
    dtreePicking_Pred["Dept"] = Depts
    dtreePicking_Pred["Date"] = actual_hour['Date']
    dtreePicking_Pred["Picking"] = dtPicking_pred
    dtreePicking_Pred['Picking_ACT'] = actual_hour['Picking']


    modelReset_dt = DecisionTreeRegressor()
    modelReset_dt.fit(train_X, train_y['Reset'])
    dtReset_pred = modelReset_dt.predict(x_test)
    dtreeReset_Pred = pd.DataFrame()
    dtreeReset_Pred["Dept"] = Depts
    dtreeReset_Pred["Date"] = actual_hour['Date']
    dtreeReset_Pred["Reset"] = dtReset_pred
    dtreeReset_Pred['Reset_ACT'] = actual_hour['Reset']


    modelBack_Offc_dt = DecisionTreeRegressor()
    modelBack_Offc_dt.fit(train_X, train_y['Back_Offc'])
    dtBack_Offc_pred = modelBack_Offc_dt.predict(x_test)
    dtreeBack_Offc_Pred = pd.DataFrame()
    dtreeBack_Offc_Pred["Dept"] = Depts
    dtreeBack_Offc_Pred["Date"] = actual_hour['Date']
    dtreeBack_Offc_Pred["Back_Offc"] = dtBack_Offc_pred
    dtreeBack_Offc_Pred['Back_Offc_ACT'] = actual_hour['Back_Offc']

    modelHR_dt = DecisionTreeRegressor()
    modelHR_dt.fit(train_X, train_y['HR'])
    dtHR_pred = modelHR_dt.predict(x_test)
    dtreeHR_Pred = pd.DataFrame()
    dtreeHR_Pred["Dept"] = Depts
    dtreeHR_Pred["Date"] = actual_hour['Date']
    dtreeHR_Pred["HR"] = dtHR_pred
    dtreeHR_Pred['HR_ACT'] = actual_hour['HR']

    modelSecurity_dt = DecisionTreeRegressor()
    modelSecurity_dt.fit(train_X, train_y['Security'])
    dtSecurity_pred = modelSecurity_dt.predict(x_test)
    dtreeSecurity_Pred = pd.DataFrame()
    dtreeSecurity_Pred["Dept"] = Depts
    dtreeSecurity_Pred["Date"] = actual_hour['Date']
    dtreeSecurity_Pred["Security"] = dtSecurity_pred
    dtreeSecurity_Pred['Security_ACT'] = actual_hour['Security']


    modelPTE_dt = DecisionTreeRegressor()
    modelPTE_dt.fit(train_X, train_y['PTE'])
    dtPTE_pred = modelPTE_dt.predict(x_test)
    dtreePTE_Pred = pd.DataFrame()
    dtreePTE_Pred["Dept"] = Depts
    dtreePTE_Pred["Date"] = actual_hour['Date']
    dtreePTE_Pred["PTE"] = dtPTE_pred
    dtreePTE_Pred['PTE_ACT'] = actual_hour['PTE']

    modelFTE_dt = DecisionTreeRegressor()
    modelFTE_dt.fit(train_X, train_y['FTE'])
    dtFTE_pred = modelFTE_dt.predict(x_test)
    dtreeFTE_Pred = pd.DataFrame()
    dtreeFTE_Pred["Dept"] = Depts
    dtreeFTE_Pred["Date"] = actual_hour['Date']
    dtreeFTE_Pred["FTE"] = dtFTE_pred
    dtreeFTE_Pred['FTE_ACT'] = actual_hour['FTE']


    Working_Hour_Pred = pd.DataFrame()
    Cleaning_Pred     = pd.DataFrame()
    Cashier_Pred      = pd.DataFrame()
    Picking_Pred      = pd.DataFrame()
    Reset_Pred        = pd.DataFrame()
    Back_Offc_Pred    = pd.DataFrame()
    HR_Pred           = pd.DataFrame()
    Security_Pred     = pd.DataFrame()
    FTE_Pred          = pd.DataFrame()
    PTE_Pred          = pd.DataFrame()
    
    
    
    Working_Hour_Pred['xgBoost']             = finaXgbWH_Pred ['Working_Hour']
    Working_Hour_Pred['dtree']               = dtree_Pred ['Working_Hour']
    Working_Hour_Pred['rf']                  = rf_Pred ['Working_Hour']
    Working_Hour_Pred['Date']                = finaXgbWH_Pred ['Date']
    #Working_Hour_Pred['Working_Hour_ACT']    = finaXgbWH_Pred['Working_Hour_ACT']
    Working_Hour_Pred = Working_Hour_Pred.sort_values('Date')
    Working_Hour_Pred = Working_Hour_Pred.reset_index(drop=True)

    Cleaning_Pred['xgBoost']                 = finaXgbCleaning_Pred ['Cleaning']
    Cleaning_Pred['dtree']                   = dtreeCleaning_Pred ['Cleaning']
    Cleaning_Pred['rf']                      = rfCleaning_Pred ['Cleaning']
    Cleaning_Pred['Date']                    = rfCleaning_Pred ['Date']
    Cleaning_Pred = Cleaning_Pred.sort_values('Date')
    Cleaning_Pred = Cleaning_Pred.reset_index(drop=True)

    Cashier_Pred['xgBoost']                  = finaXgbCashier_Pred ['Cashier']
    Cashier_Pred['dtree']                    = dtreeCashier_Pred ['Cashier']
    Cashier_Pred['rf']                       = rfCashier_Pred ['Cashier']
    Cashier_Pred['Date']                    = rfCashier_Pred ['Date']
    Cashier_Pred = Cashier_Pred.sort_values('Date')
    Cashier_Pred = Cashier_Pred.reset_index(drop=True)

    Picking_Pred['xgBoost']                  = finaXgbPicking_Pred ['Picking']
    Picking_Pred['dtree']                    = dtreePicking_Pred ['Picking']
    Picking_Pred['rf']                       = rfPicking_Pred ['Picking']
    Picking_Pred['Date']                     = rfPicking_Pred ['Date']
    Picking_Pred = Picking_Pred.sort_values('Date')
    Picking_Pred = Picking_Pred.reset_index(drop=True)

    Reset_Pred['xgBoost']                    = finaXgbReset_Pred ['Reset']
    Reset_Pred['dtree']                      = dtreeReset_Pred ['Reset']
    Reset_Pred['rf']                         = rfReset_Pred ['Reset']
    Reset_Pred['Date']                       = rfReset_Pred ['Date']
    Reset_Pred = Reset_Pred.sort_values('Date')
    Reset_Pred = Reset_Pred.reset_index(drop=True)

    Back_Offc_Pred['xgBoost']                = finaXgbBackOffc_Pred ['Back_Offc']
    Back_Offc_Pred['dtree']                  = dtreeBack_Offc_Pred ['Back_Offc']
    Back_Offc_Pred['rf']                     = rfBack_Offc_Pred ['Back_Offc']
    Back_Offc_Pred['Date']                   = rfBack_Offc_Pred ['Date']
    Back_Offc_Pred = Back_Offc_Pred.sort_values('Date')
    Back_Offc_Pred = Back_Offc_Pred.reset_index(drop=True)

    HR_Pred['xgBoost']                       = finaXgbHR_Pred ['HR']
    HR_Pred['dtree']                         = dtreeHR_Pred ['HR']
    HR_Pred['rf']                            = rfHR_Pred ['HR']
    HR_Pred['Date']                          = rfHR_Pred ['Date']
    HR_Pred = HR_Pred.sort_values('Date')
    HR_Pred = HR_Pred.reset_index(drop=True)

    Security_Pred['xgBoost']                 = finaXgbSecurity_Pred ['Security']
    Security_Pred['dtree']                   = dtreeSecurity_Pred ['Security']
    Security_Pred['rf']                      = rfSecurity_Pred ['Security']
    Security_Pred['Date']                    = rfSecurity_Pred ['Date']
    Security_Pred = Security_Pred.sort_values('Date')
    Security_Pred = Security_Pred.reset_index(drop=True)

    FTE_Pred['xgBoost']                      = finaXgb_Pred ['FTE']
    FTE_Pred['dtree']                        = dtreeFTE_Pred ['FTE']
    FTE_Pred['rf']                           = rfFTE_Pred ['FTE']
    FTE_Pred['Date']                         = rfFTE_Pred ['Date']
    FTE_Pred = FTE_Pred.sort_values('Date')
    FTE_Pred = FTE_Pred.reset_index(drop=True)

    PTE_Pred['xgBoost']                      = finaXgbPTE_Pred ['PTE']
    PTE_Pred['dtree']                        = dtreePTE_Pred ['PTE']
    PTE_Pred['rf']                           = rfPTE_Pred ['PTE']
    PTE_Pred['Date']                         = rfPTE_Pred ['Date']
    PTE_Pred = PTE_Pred.sort_values('Date')
    PTE_Pred = PTE_Pred.reset_index(drop=True)

    
    actual_hour = actual_hour.sort_values('Date')
    actual_hour = actual_hour.reset_index(drop=True)
    #actual_hour
    #a = pd.unique(actual_hour['Department'])
    #a[0]
    
    
    all_Activities = actual_hour.columns;
    #all_Activities
    
    """
    WH_FTE_PTE = pd.DataFrame()
    act_Total_Working_Hour = actual_hour['Working_Hour']    
    WH_FTE_PTE = actual_hour[['Working_Hour','FTE','PTE']]
    WH_FTE_PTE_column = WH_FTE_PTE.columns
    actual_hour = actual_hour.drop(['Department','Working_Hour','FTE','PTE'],axis=1)
   
    #Reset_Pred[Reset_Pred.index == selected_dept][selected_model]

"""
    bak_actual_hour = actual_hour
    bak_actual_hour     = bak_actual_hour.drop(['Department','Working_Hour','FTE','PTE','Date'],axis=1)
    #final_xgb_pred = final_xgb_prediction[['Cleaning','Cashier','Picking','Reset','Back_Offc','HR','Security']].sum()

    final_xgb_prediction = pd.DataFrame()
    selected_model = 'xgBoost'
    final_xgb_prediction['Date']          = actual_hour ['Date']
    final_xgb_prediction['Cleaning'] = Cleaning_Pred[selected_model]
    final_xgb_prediction['Cashier'] = Cashier_Pred[selected_model]
    final_xgb_prediction['Picking'] = Picking_Pred[selected_model]
    final_xgb_prediction['Reset'] = Reset_Pred[selected_model]
    final_xgb_prediction['Back_Offc'] = Back_Offc_Pred[selected_model]
    final_xgb_prediction['Service Desk'] = HR_Pred[selected_model]
    final_xgb_prediction['Security'] = Security_Pred[selected_model]

    final_dtree_prediction = pd.DataFrame()
    selected_model = 'dtree'
    final_dtree_prediction['Date']          = actual_hour ['Date']
    final_dtree_prediction['Cleaning'] = Cleaning_Pred[selected_model]
    final_dtree_prediction['Cashier'] = Cashier_Pred[selected_model]
    final_dtree_prediction['Picking'] = Picking_Pred[selected_model]
    final_dtree_prediction['Reset'] = Reset_Pred[selected_model]
    final_dtree_prediction['Back_Offc'] = Back_Offc_Pred[selected_model]
    final_dtree_prediction['Service Desk'] = HR_Pred[selected_model]
    final_dtree_prediction['Security'] = Security_Pred[selected_model]


    final_rf_prediction = pd.DataFrame()
    selected_model = 'rf'
    final_rf_prediction['Date']          = actual_hour ['Date']
    final_rf_prediction['Cleaning'] = Cleaning_Pred[selected_model]
    final_rf_prediction['Cashier'] = Cashier_Pred[selected_model]
    final_rf_prediction['Picking'] = Picking_Pred[selected_model]
    final_rf_prediction['Reset'] = Reset_Pred[selected_model]
    final_rf_prediction['Back_Offc'] = Back_Offc_Pred[selected_model]
    final_rf_prediction['Service Desk'] = HR_Pred[selected_model]
    final_rf_prediction['Security'] = Security_Pred[selected_model]

    final_WH_PT_FT_prediction = pd.DataFrame()
    selected_model = 'xgBoost'
    final_WH_PT_FT_prediction['Date']          = actual_hour ['Date']
    final_WH_PT_FT_prediction['Working_Hour'] = Working_Hour_Pred[selected_model]
    final_WH_PT_FT_prediction['FTE'] = FTE_Pred[selected_model]
    final_WH_PT_FT_prediction['PTE'] = PTE_Pred[selected_model]


    final_WH_PT_FT_rf_prediction = pd.DataFrame()
    selected_model = 'rf'
    final_WH_PT_FT_rf_prediction['Date']          = actual_hour ['Date']
    final_WH_PT_FT_rf_prediction['Working_Hour'] = Working_Hour_Pred[selected_model]
    final_WH_PT_FT_rf_prediction['FTE'] = FTE_Pred[selected_model]
    final_WH_PT_FT_rf_prediction['PTE'] = PTE_Pred[selected_model]

    final_WH_PT_FT_dtree_prediction = pd.DataFrame()
    selected_model = 'dtree'
    final_WH_PT_FT_dtree_prediction['Date']          = actual_hour ['Date']
    final_WH_PT_FT_dtree_prediction['Working_Hour'] = Working_Hour_Pred[selected_model]
    final_WH_PT_FT_dtree_prediction['FTE'] = FTE_Pred[selected_model]
    final_WH_PT_FT_dtree_prediction['PTE'] = PTE_Pred[selected_model]
    
    final_WH_PT_FT_prediction_bak = final_WH_PT_FT_prediction
    final_WH_PT_FT_prediction     = final_WH_PT_FT_prediction.drop(['Date'],axis=1)
    #final_WH_PT_FT_prediction

    final_WH_PT_FT_rf_prediction_bak = final_WH_PT_FT_rf_prediction
    final_WH_PT_FT_rf_prediction     = final_WH_PT_FT_rf_prediction.drop(['Date'],axis=1)
    #final_WH_PT_FT_rf_prediction

    final_WH_PT_FT_dtree_prediction_bak = final_WH_PT_FT_dtree_prediction
    final_WH_PT_FT_dtree_prediction           = final_WH_PT_FT_dtree_prediction.drop(['Date'],axis=1)
    #final_WH_PT_FT_dtree_prediction
    
    
    final_WH_PT_FT_dtree_pred = pd.DataFrame()
    final_WH_PT_FT_dtree_pred = final_WH_PT_FT_dtree_prediction[['Working_Hour','FTE','PTE']].sum()
    #final_WH_PT_FT_dtree_pred
    
    final_WH_PT_FT_rf_pred = pd.DataFrame()
    final_WH_PT_FT_rf_pred = final_WH_PT_FT_rf_prediction[['Working_Hour','FTE','PTE']].sum()
    #final_WH_PT_FT_rf_pred

    final_WH_PT_FT_pred = pd.DataFrame()
    final_WH_PT_FT_pred = final_WH_PT_FT_prediction[['Working_Hour','FTE','PTE']].sum()
    #final_WH_PT_FT_pred
    
    
    final_xgb_pred_w_date = final_xgb_prediction
    final_xgb_prediction  = final_xgb_prediction.drop(['Date'],axis=1)
    #final_xgb_prediction
    
    final_dtree_pred_w_date = final_dtree_prediction
    final_dtree_prediction  = final_dtree_prediction.drop(['Date'],axis=1)
    #final_dtree_prediction
    
    final_rf_pred_w_date = final_rf_prediction
    final_rf_prediction  = final_rf_prediction.drop(['Date'],axis=1)
    final_rf_prediction

    final_actual_hour_w_date = actual_hour
    actual_hour = actual_hour.drop(['Date','Department','FTE','PTE','Working_Hour'],axis=1)
    actual_hour

    final_xgb_pred = pd.DataFrame()
    final_xgb_pred = final_xgb_prediction[['Cleaning','Cashier','Picking','Reset','Back_Offc','Service Desk','Security']].sum()
    final_xgb_pred

    final_dtree_pred = pd.DataFrame()
    final_dtree_pred = final_dtree_prediction[['Cleaning','Cashier','Picking','Reset','Back_Offc','Service Desk','Security']].sum()
    final_dtree_pred

    final_rf_pred = pd.DataFrame()
    final_rf_pred = final_rf_prediction[['Cleaning','Cashier','Picking','Reset','Back_Offc','Service Desk','Security']].sum()
    final_rf_pred
    
    actual_hour=actual_hour.rename(columns = {'HR':'Service Desk'})
    actual_hour_pred = pd.DataFrame()
    actual_hour_pred = actual_hour[['Cleaning','Cashier','Picking','Reset','Back_Offc','Service Desk','Security']].sum()
    actual_hour_pred

    WH_FTE_PTE = pd.DataFrame()
    all_date = final_actual_hour_w_date['Date']    
    WH_FTE_PTE = final_actual_hour_w_date[['Working_Hour','FTE','PTE']]
    WH_FTE_PTE_column = WH_FTE_PTE.columns
    #actual_hour = actual_hour.drop(['Department','Working_Hour','FTE','PTE'],axis=1)
    
    actual_hour_columns =   actual_hour.columns
    #WH_FTE_PTE
    #df['MyColumn'].sum()
    
    WH_FTE_PTE_PRED = pd.DataFrame()
    WH_FTE_PTE_PRED = WH_FTE_PTE[['Working_Hour','FTE','PTE']].sum()
    #WH_FTE_PTE_PRED
   # create_dash_application(app,final_xgb_pred_w_date,final_dtree_pred_w_date,final_rf_pred_w_date,final_actual_hour_w_date)        
    if(futurePred  == True):
        if((Individual == 'Total_hour') ):    
            fig = go.Figure(data=[
            go.Bar(name='XGboost Predicted', x=actual_hour_columns, y=final_xgb_pred, marker_color ='#12a4d9' , text=final_xgb_pred,),
            go.Bar(name='Random forest Predicted', x=actual_hour_columns, y=final_rf_pred,marker_color ='#322e2f', text=final_rf_pred ,),    
            go.Bar(name='Descision Tree Predicted', x=actual_hour_columns, y=final_dtree_pred, marker_color ='#e2d810', text=final_dtree_pred,),
            #go.Bar(name='Actual ', x=actual_hour_columns, y=actual_hour_pred,marker_color ='#322e2f',text=actual_hour_pred),
            ])
            fig.update_layout(barmode='group',)
            fig.update_layout(
                template="plotly_white",
                xaxis=dict(title_text="Roles"),
                yaxis=dict(title_text="Working Hour"),
                bargroupgap=0.1,
                bargap=0.3,
                width = 1200,
                #title_text = '<a href = "/role"> click details</a>'
                title_text = "Workforce prediction for All roles for store :"+storeChoosen+', Departmnet : '+deptChoosen+'  '+dateRangeText+' '+'<a href = "/role">  Details</a>'
            )
            fig.update_traces(texttemplate='<b>%{text:.3s}<b>', textposition='outside')
            #fig.update_traces(title_text = '<a href = "#"> click details</a>')
            #fig.show()
            plot_div = plot(fig,output_type='div')    
            #print (' Error ')
            
            fig = go.Figure()
            fig.update_layout(
                template="plotly_white",
                xaxis=dict(title_text="Roles"),
                yaxis=dict(title_text="Working hour"),
                width = 1200,
                title_text = "Workforce prediction for All roles for store :"+storeChoosen+', Departmnet : '+deptChoosen+'  '+dateRangeText+' '+'<a href = "/role">  Details</a>'

            )
            #fig.add_trace(go.Scatter(x=actual_hour_columns, y=actual_hour_pred,
             #                   mode='lines+markers+text',
             #                   name='Actual',
              #                  text=actual_hour_pred))
            fig.add_trace(go.Scatter(x=actual_hour_columns, y=final_xgb_pred,
                                mode='lines+markers+text',
                                name='XGBoost Predicted',
                                text=final_xgb_pred)) 
            fig.add_trace(go.Scatter(x=actual_hour_columns, y=final_dtree_pred,
                                mode='lines+markers',
                                name='DTree Predicted',
                                text=final_dtree_pred,
                                ))
            fig.add_trace(go.Scatter(x=actual_hour_columns, y=final_rf_pred,
                                mode='lines+markers',
                                name='Random Forest Predicted',
                                text=final_rf_pred,
                                ))  
            fig.update_traces(texttemplate='%{text:.3s}', textposition='top center') 
            plot_div1 = plot(fig,output_type='div')   
            
        else:
            multiRole = multiRole
            print(multiRole)
            actual_hour_columns = actual_hour[multiRole].columns
            actual_hour_pred =actual_hour_pred[multiRole]
            final_xgb_pred=final_xgb_pred[multiRole]
            final_rf_pred=final_rf_pred[multiRole]
            final_dtree_pred=final_dtree_pred[multiRole]
            
            fig = go.Figure(data=[
            go.Bar(name='XGboost Predicted', x=actual_hour_columns, y=final_xgb_pred, marker_color ='#12a4d9' , text=final_xgb_pred,),
            go.Bar(name='Random forest Predicted', x=actual_hour_columns, y=final_rf_pred,marker_color ='#322e2f', text=final_rf_pred ,),    
            go.Bar(name='Descision Tree Predicted', x=actual_hour_columns, y=final_dtree_pred, marker_color ='#e2d810', text=final_dtree_pred,),
           # go.Bar(name='Actual ', x=actual_hour_columns, y=actual_hour_pred,marker_color ='#322e2f',text=actual_hour_pred),
            ])
            fig.update_layout(barmode='group',)
            fig.update_layout(
                template="plotly_white",
                xaxis=dict(title_text="Roles"),
                yaxis=dict(title_text="Working Hour"),
                bargroupgap=0.1,
                bargap=0.3,
                width = 1100,
                #title_text = '<a href = "/role"> click details</a>'
                title_text = "Workforce prediction for different roles for store :"+storeChoosen+', Departmnet : '+deptChoosen+'  '+dateRangeText+' '+'<a href = "/role">  Details</a>'
            )
            fig.update_traces(texttemplate='<b>%{text:.3s}<b>', textposition='outside')
            #fig.update_traces(title_text = '<a href = "#"> click details</a>')
            #fig.show()
            plot_div = plot(fig,output_type='div')
            
            
            fig = go.Figure()
            fig.update_layout(
                template="plotly_white",
                xaxis=dict(title_text="Roles"),
                yaxis=dict(title_text="Working hour"),
                width = 1200,
                title_text = "Workforce prediction for All roles for store :"+storeChoosen+', Departmnet : '+deptChoosen+'  '+dateRangeText+' '+'<a href = "/role"> click details</a>'

            )
            #fig.add_trace(go.Scatter(x=actual_hour_columns, y=actual_hour_pred,
             #                   mode='lines+markers+text',
             #                   name='Actual',
             #                   text=actual_hour_pred))
            fig.add_trace(go.Scatter(x=actual_hour_columns, y=final_xgb_pred,
                                mode='lines+markers+text',
                                name='XGBoost Predicted',
                                text=final_xgb_pred))
            fig.add_trace(go.Scatter(x=actual_hour_columns, y=final_dtree_pred,
                                mode='lines+markers',
                                name='DTree Predicted',
                                text=final_dtree_pred,
                               ))
            fig.add_trace(go.Scatter(x=actual_hour_columns, y=final_rf_pred,
                                mode='lines+markers',
                                name='Random Forest Predicted',
                                text=final_rf_pred,
                               ))  
            fig.update_traces(texttemplate='%{text:.3s}', textposition='top center')    
            plot_div1 = plot(fig,output_type='div')   
                
            #print (multiRole)
            
        if((FTE_value == None) & (PTE_var == None)):
            print ('Both None')
        elif((FTE_value != None) & (PTE_var == None)):
            WH_FTE_PTE_column         = WH_FTE_PTE[['FTE']].columns
            final_WH_PT_FT_pred       = final_WH_PT_FT_pred[['FTE']]
            final_WH_PT_FT_rf_pred    = final_WH_PT_FT_rf_pred[['FTE']]
            final_WH_PT_FT_dtree_pred = final_WH_PT_FT_dtree_pred[['FTE']]
            WH_FTE_PTE_PRED           = WH_FTE_PTE_PRED[['FTE']]
            fig = go.Figure(data=[
                go.Bar(name='XGboost Predicted', x=WH_FTE_PTE_column, y=final_WH_PT_FT_pred, marker_color ='#1e3d59' , text=final_WH_PT_FT_pred,width = 0.08),
                go.Bar(name='Random forest Predicted', x=WH_FTE_PTE_column, y=final_WH_PT_FT_rf_pred,marker_color ='#f5f0e1',text= final_WH_PT_FT_rf_pred,width = 0.08),    
                go.Bar(name='Descision Tree Predicted', x=WH_FTE_PTE_column, y=final_WH_PT_FT_dtree_pred, marker_color ='#ff6e40',text=final_WH_PT_FT_dtree_pred,width = 0.08),
                #go.Bar(name='Actual ', x=WH_FTE_PTE_column, y=WH_FTE_PTE_PRED,marker_color ='#ffc13b',text=WH_FTE_PTE_PRED,width = 0.08),
            ])
            fig.update_layout(barmode='group')
            fig.update_layout(
                template="plotly_white",
                xaxis=dict(title_text="Employeement Type"),
                yaxis=dict(title_text="Working Hour"),
                width = 1100,
                height = 480,
                bargroupgap=0.1,
                bargap=0.3,
                #title_text = "Total Workforce prediction  for the Department with Employeement type :"
                title_text = "Total Workforce prediction for each Employment Type for store :"+storeChoosen+', Departmnet : '+deptChoosen +'  '+dateRangeText
                
            )
            fig.update_traces(texttemplate='<b>%{text:.3s}<b>', textposition='outside')
            plot_div2 = plot(fig,output_type='div')
            print ('FTE value is not None PTE_var none')
        elif((FTE_value == None) & (PTE_var != None)):
            WH_FTE_PTE_column         = WH_FTE_PTE[['PTE']].columns
            final_WH_PT_FT_pred       = final_WH_PT_FT_pred[['PTE']]
            final_WH_PT_FT_rf_pred    = final_WH_PT_FT_rf_pred[['PTE']]
            final_WH_PT_FT_dtree_pred = final_WH_PT_FT_dtree_pred[['PTE']]
            WH_FTE_PTE_PRED           = WH_FTE_PTE_PRED[['PTE']]
            print ('FTE value is  None PTE_var not none')
            fig = go.Figure(data=[
                go.Bar(name='XGboost Predicted', x=WH_FTE_PTE_column, y=final_WH_PT_FT_pred, marker_color ='#1e3d59' , text=final_WH_PT_FT_pred,width = 0.08),
                go.Bar(name='Random forest Predicted', x=WH_FTE_PTE_column, y=final_WH_PT_FT_rf_pred,marker_color ='#f5f0e1',text= final_WH_PT_FT_rf_pred,width = 0.08),    
                go.Bar(name='Descision Tree Predicted', x=WH_FTE_PTE_column, y=final_WH_PT_FT_dtree_pred, marker_color ='#ff6e40',text=final_WH_PT_FT_dtree_pred,width = 0.08),
                #go.Bar(name='Actual ', x=WH_FTE_PTE_column, y=WH_FTE_PTE_PRED,marker_color ='#ffc13b',text=WH_FTE_PTE_PRED,width = 0.08),
            ])
            fig.update_layout(barmode='group')
            fig.update_layout(
                template="plotly_white",
                xaxis=dict(title_text="Employeement Type"),
                yaxis=dict(title_text="Working Hour"),
                width = 1100,
                height = 480,
                bargroupgap=0.1,
                bargap=0.3,
                #title_text = "Total Workforce prediction  for the Department with Employeement type :"
                title_text = "Total Workforce prediction for each Employment Type for store :"+storeChoosen+', Departmnet : '+deptChoosen +'  '+dateRangeText
                
            )
            fig.update_traces(texttemplate='<b>%{text:.3s}<b>', textposition='outside')
            plot_div2 = plot(fig,output_type='div')
        else:
            print ('print ALL')
            fig = go.Figure(data=[
                go.Bar(name='XGboost Predicted', x=WH_FTE_PTE_column, y=final_WH_PT_FT_pred, marker_color ='#1e3d59' , text=final_WH_PT_FT_pred),
                go.Bar(name='Random forest Predicted', x=WH_FTE_PTE_column, y=final_WH_PT_FT_rf_pred,marker_color ='#f5f0e1',text= final_WH_PT_FT_rf_pred),    
                go.Bar(name='Descision Tree Predicted', x=WH_FTE_PTE_column, y=final_WH_PT_FT_dtree_pred, marker_color ='#ff6e40',text=final_WH_PT_FT_dtree_pred),
                #go.Bar(name='Actual ', x=WH_FTE_PTE_column, y=WH_FTE_PTE_PRED,marker_color ='#ffc13b',text=WH_FTE_PTE_PRED),
            ])
            fig.update_layout(barmode='group')
            fig.update_layout(
                template="plotly_white",
                xaxis=dict(title_text="Employeement Type"),
                yaxis=dict(title_text="Working Hour"),
                width = 1100,
                bargroupgap=0.1,
                height = 480,
                bargap=0.3,
                #title_text = "Total Workforce prediction  for the Department with Employeement type :"
                title_text = "Total Workforce prediction for each Employment Type for store :"+storeChoosen+', Departmnet : '+deptChoosen +'  '+dateRangeText
                
            )
            fig.update_traces(texttemplate='<b>%{text:.3s}<b>', textposition='outside')
            plot_div2 = plot(fig,output_type='div')

    else:        
        if((Individual == 'Total_hour') ):    
            fig = go.Figure(data=[
            go.Bar(name='XGboost Predicted', x=actual_hour_columns, y=final_xgb_pred, marker_color ='#12a4d9' , text=final_xgb_pred,),
            go.Bar(name='Random forest Predicted', x=actual_hour_columns, y=final_rf_pred,marker_color ='#d9138a', text=final_rf_pred ,visible='legendonly'),    
            go.Bar(name='Descision Tree Predicted', x=actual_hour_columns, y=final_dtree_pred, marker_color ='#e2d810', text=final_dtree_pred,visible='legendonly'),
            go.Bar(name='Actual ', x=actual_hour_columns, y=actual_hour_pred,marker_color ='#322e2f',text=actual_hour_pred),
            ])
            fig.update_layout(barmode='group',)
            fig.update_layout(
                template="plotly_white",
                xaxis=dict(title_text="Roles"),
                yaxis=dict(title_text="Working Hour"),
                bargroupgap=0.1,
                bargap=0.3,
                width = 1200,
                #title_text = '<a href = "/role"> click details</a>'
                title_text = "Workforce prediction for All roles for store :"+storeChoosen+', Departmnet : '+deptChoosen+'  '+dateRangeText+' '+'<a href = "/role">  Details</a>'
            )
            fig.update_traces(texttemplate='<b>%{text:.3s}<b>', textposition='outside')
            #fig.update_traces(title_text = '<a href = "#"> click details</a>')
            #fig.show()
            plot_div = plot(fig,output_type='div')    
            #print (' Error ')
            
            fig = go.Figure()
            fig.update_layout(
                template="plotly_white",
                xaxis=dict(title_text="Roles"),
                yaxis=dict(title_text="Working hour"),
                width = 1200,
                title_text = "Workforce prediction for All roles for store :"+storeChoosen+', Departmnet : '+deptChoosen+'  '+dateRangeText+' '+'<a href = "/role">  Details</a>'

            )
            fig.add_trace(go.Scatter(x=actual_hour_columns, y=actual_hour_pred,
                                mode='lines+markers+text',
                                name='Actual',
                                text=actual_hour_pred))
            fig.add_trace(go.Scatter(x=actual_hour_columns, y=final_xgb_pred,
                                mode='lines+markers+text',
                                name='XGBoost Predicted',
                                text=final_xgb_pred))
            fig.add_trace(go.Scatter(x=actual_hour_columns, y=final_dtree_pred,
                                mode='lines+markers',
                                name='DTree Predicted',
                                text=final_dtree_pred,
                                visible='legendonly'))
            fig.add_trace(go.Scatter(x=actual_hour_columns, y=final_rf_pred,
                                mode='lines+markers',
                                name='Random Forest Predicted',
                                text=final_rf_pred,
                                visible='legendonly'))  
            fig.update_traces(texttemplate='%{text:.3s}', textposition='top center') 
            plot_div1 = plot(fig,output_type='div')   
            
        else:
            multiRole = multiRole
            print(multiRole)
            actual_hour_columns = actual_hour[multiRole].columns
            actual_hour_pred =actual_hour_pred[multiRole]
            final_xgb_pred=final_xgb_pred[multiRole]
            final_rf_pred=final_rf_pred[multiRole]
            final_dtree_pred=final_dtree_pred[multiRole]
            
            fig = go.Figure(data=[
            go.Bar(name='XGboost Predicted', x=actual_hour_columns, y=final_xgb_pred, marker_color ='#12a4d9' , text=final_xgb_pred,),
            go.Bar(name='Random forest Predicted', x=actual_hour_columns, y=final_rf_pred,marker_color ='#d9138a', text=final_rf_pred ,visible='legendonly'),    
            go.Bar(name='Descision Tree Predicted', x=actual_hour_columns, y=final_dtree_pred, marker_color ='#e2d810', text=final_dtree_pred,),
            go.Bar(name='Actual ', x=actual_hour_columns, y=actual_hour_pred,marker_color ='#322e2f',text=actual_hour_pred),
            ])
            fig.update_layout(barmode='group',)
            fig.update_layout(
                template="plotly_white",
                xaxis=dict(title_text="Roles"),
                yaxis=dict(title_text="Working Hour"),
                bargroupgap=0.1,
                bargap=0.3,
                width = 1100,
                #title_text = '<a href = "/role"> click details</a>'
                title_text = "Workforce prediction for different roles for store :"+storeChoosen+', Departmnet : '+deptChoosen+'  '+dateRangeText+' '+'<a href = "/role">  Details</a>'
            )
            fig.update_traces(texttemplate='<b>%{text:.3s}<b>', textposition='outside')
            #fig.update_traces(title_text = '<a href = "#"> click details</a>')
            #fig.show()
            plot_div = plot(fig,output_type='div')
            
            
            fig = go.Figure()
            fig.update_layout(
                template="plotly_white",
                xaxis=dict(title_text="Roles"),
                yaxis=dict(title_text="Working hour"),
                width = 1200,
                title_text = "Workforce prediction for All roles for store :"+storeChoosen+', Departmnet : '+deptChoosen+'  '+dateRangeText+' '+'<a href = "/role"> click details</a>'

            )
            fig.add_trace(go.Scatter(x=actual_hour_columns, y=actual_hour_pred,
                                mode='lines+markers+text',
                                name='Actual',
                                text=actual_hour_pred))
            fig.add_trace(go.Scatter(x=actual_hour_columns, y=final_xgb_pred,
                                mode='lines+markers+text',
                                name='XGBoost Predicted',
                                text=final_xgb_pred))
            fig.add_trace(go.Scatter(x=actual_hour_columns, y=final_dtree_pred,
                                mode='lines+markers',
                                name='DTree Predicted',
                                text=final_dtree_pred,
                                visible='legendonly'))
            fig.add_trace(go.Scatter(x=actual_hour_columns, y=final_rf_pred,
                                mode='lines+markers',
                                name='Random Forest Predicted',
                                text=final_rf_pred,
                                visible='legendonly'))  
            fig.update_traces(texttemplate='%{text:.3s}', textposition='top center')    
            plot_div1 = plot(fig,output_type='div')   
                
            #print (multiRole)
            
        if((FTE_value == None) & (PTE_var == None)):
            print ('Both None')
        elif((FTE_value != None) & (PTE_var == None)):
            WH_FTE_PTE_column         = WH_FTE_PTE[['FTE']].columns
            final_WH_PT_FT_pred       = final_WH_PT_FT_pred[['FTE']]
            final_WH_PT_FT_rf_pred    = final_WH_PT_FT_rf_pred[['FTE']]
            final_WH_PT_FT_dtree_pred = final_WH_PT_FT_dtree_pred[['FTE']]
            WH_FTE_PTE_PRED           = WH_FTE_PTE_PRED[['FTE']]
            fig = go.Figure(data=[
                go.Bar(name='XGboost Predicted', x=WH_FTE_PTE_column, y=final_WH_PT_FT_pred, marker_color ='#1e3d59' , text=final_WH_PT_FT_pred,width = 0.08),
                go.Bar(name='Random forest Predicted', x=WH_FTE_PTE_column, y=final_WH_PT_FT_rf_pred,marker_color ='#f5f0e1',text= final_WH_PT_FT_rf_pred,width = 0.08),    
                go.Bar(name='Descision Tree Predicted', x=WH_FTE_PTE_column, y=final_WH_PT_FT_dtree_pred, marker_color ='#ff6e40',text=final_WH_PT_FT_dtree_pred,width = 0.08),
                go.Bar(name='Actual ', x=WH_FTE_PTE_column, y=WH_FTE_PTE_PRED,marker_color ='#ffc13b',text=WH_FTE_PTE_PRED,width = 0.08),
            ])
            fig.update_layout(barmode='group')
            fig.update_layout(
                template="plotly_white",
                xaxis=dict(title_text="Employeement Type"),
                yaxis=dict(title_text="Working Hour"),
                width = 1100,
                height = 480,
                bargroupgap=0.1,
                bargap=0.3,
                #title_text = "Total Workforce prediction  for the Department with Employeement type :"
                title_text = "Total Workforce prediction for each Employment Type for store :"+storeChoosen+', Departmnet : '+deptChoosen +'  '+dateRangeText
                
            )
            fig.update_traces(texttemplate='<b>%{text:.3s}<b>', textposition='outside')
            plot_div2 = plot(fig,output_type='div')
            print ('FTE value is not None PTE_var none')
        elif((FTE_value == None) & (PTE_var != None)):
            WH_FTE_PTE_column         = WH_FTE_PTE[['PTE']].columns
            final_WH_PT_FT_pred       = final_WH_PT_FT_pred[['PTE']]
            final_WH_PT_FT_rf_pred    = final_WH_PT_FT_rf_pred[['PTE']]
            final_WH_PT_FT_dtree_pred = final_WH_PT_FT_dtree_pred[['PTE']]
            WH_FTE_PTE_PRED           = WH_FTE_PTE_PRED[['PTE']]
            print ('FTE value is  None PTE_var not none')
            fig = go.Figure(data=[
                go.Bar(name='XGboost Predicted', x=WH_FTE_PTE_column, y=final_WH_PT_FT_pred, marker_color ='#1e3d59' , text=final_WH_PT_FT_pred,width = 0.08),
                go.Bar(name='Random forest Predicted', x=WH_FTE_PTE_column, y=final_WH_PT_FT_rf_pred,marker_color ='#f5f0e1',text= final_WH_PT_FT_rf_pred,width = 0.08),    
                go.Bar(name='Descision Tree Predicted', x=WH_FTE_PTE_column, y=final_WH_PT_FT_dtree_pred, marker_color ='#ff6e40',text=final_WH_PT_FT_dtree_pred,width = 0.08),
                go.Bar(name='Actual ', x=WH_FTE_PTE_column, y=WH_FTE_PTE_PRED,marker_color ='#ffc13b',text=WH_FTE_PTE_PRED,width = 0.08),
            ])
            fig.update_layout(barmode='group')
            fig.update_layout(
                template="plotly_white",
                xaxis=dict(title_text="Employeement Type"),
                yaxis=dict(title_text="Working Hour"),
                width = 1100,
                height = 480,
                bargroupgap=0.1,
                bargap=0.3,
                #title_text = "Total Workforce prediction  for the Department with Employeement type :"
                title_text = "Total Workforce prediction for each Employment Type for store :"+storeChoosen+', Departmnet : '+deptChoosen +'  '+dateRangeText
                
            )
            fig.update_traces(texttemplate='<b>%{text:.3s}<b>', textposition='outside')
            plot_div2 = plot(fig,output_type='div')
        else:
            print ('print ALL')
            fig = go.Figure(data=[
                go.Bar(name='XGboost Predicted', x=WH_FTE_PTE_column, y=final_WH_PT_FT_pred, marker_color ='#1e3d59' , text=final_WH_PT_FT_pred),
                go.Bar(name='Random forest Predicted', x=WH_FTE_PTE_column, y=final_WH_PT_FT_rf_pred,marker_color ='#f5f0e1',text= final_WH_PT_FT_rf_pred),    
                go.Bar(name='Descision Tree Predicted', x=WH_FTE_PTE_column, y=final_WH_PT_FT_dtree_pred, marker_color ='#ff6e40',text=final_WH_PT_FT_dtree_pred),
                go.Bar(name='Actual ', x=WH_FTE_PTE_column, y=WH_FTE_PTE_PRED,marker_color ='#ffc13b',text=WH_FTE_PTE_PRED),
            ])
            fig.update_layout(barmode='group')
            fig.update_layout(
                template="plotly_white",
                xaxis=dict(title_text="Employeement Type"),
                yaxis=dict(title_text="Working Hour"),
                width = 1100,
                bargroupgap=0.1,
                height = 480,
                bargap=0.3,
                #title_text = "Total Workforce prediction  for the Department with Employeement type :"
                title_text = "Total Workforce prediction for each Employment Type for store :"+storeChoosen+', Departmnet : '+deptChoosen +'  '+dateRangeText
                
            )
            fig.update_traces(texttemplate='<b>%{text:.3s}<b>', textposition='outside')
            plot_div2 = plot(fig,output_type='div')
    return plot_div+plot_div1+plot_div2

    #return render(request,"default.html",context={'plot_div': plot_div},)
    #return render_template("default.html")

def missing_values_table(data):       
        mis_val = data.isnull().sum()                                     # Total missing values         
        mis_val_percent = 100 * data.isnull().sum() / len(data)           # Percentage of missing values        
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)     # Make a table with the results        
        mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})   # Rename the columns      
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)         
        print ("Your selected dataframe has " + str(data.shape[1]) + " columns.\nThere are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")  
        return mis_val_table_ren_columns
    

if __name__ == "__main__":
    app.run(debug=True , port=8081)