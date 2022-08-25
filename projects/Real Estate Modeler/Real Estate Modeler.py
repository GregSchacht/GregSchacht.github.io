import numpy as np
import pandas as pd
from itertools import product
import os
import keyboard
import datetime
import re

def get_model_name(e) :
    mn = str(e).split("(")[0]

    if 'catboost' in str(e):
        mn = 'CatBoostClassifier'
    
    model_dict_logging = {"LinearRegression" : "lr",
                          'BayesianRidge':"br",
                          'Ridge':"ridge",
                          'OrthogonalMatchingPursuit':"omp",
                          'DecisionTreeRegressor':"dt",
                          'KNeighborsRegressor':"knn",
                          'PassiveAggressiveRegressor':"par",
                          'ElasticNet':"en",
                          'Lasso':"lasso",
                          'LassoLars':"llar",
                          'RandomSampleConsensus':"ransac",
                          'SVR': "svm"} 

    return model_dict_logging[mn]

def anchordate(df, date):
    #We want to create a list of "anchor date values" to express numbers as to normalize disparate median values
    reg=re.compile(".+[MmYy/]{2}.*") #Use Regex to remove all the M/M and Y/Y values, we don't care about those for this transformation
    drops=["cbsa_title", "HouseholdRank"]
    for col, vals in df.iteritems(): #Find all the M/M and Y/Y values, and add them to the list to be dropped
        if reg.search(col) != None:
            drops=drops+[reg.search(col).group()]
    #Now we need to create a dataframe upon which we can operate to transform all the values
    holdf=df.drop(drops, axis=1).set_index("Date").astype(float) #Make sortable by CBSA by swapping index layers
    holdf["cbsa_code"]=holdf["cbsa_code"].astype(int)
    startvals=df.loc[df["Date"]=="2017-07-01"].drop(drops, axis=1).set_index("Date").astype(float) #Grab a snapshot of our anchor date
    startvals["cbsa_code"]=startvals["cbsa_code"].astype(int)
    holdf=holdf.reset_index().set_index(["Date","cbsa_code"]).divide(startvals.set_index("cbsa_code")).sort_values(by=["Date", "cbsa_code"]) #Divide each value inside by its corresponding CBSA in startvals
    rename={}
    for i in holdf.columns: #Rename all the columns to reflect that they're now pct diff
        if i != "Date":
            rename.update({i:i.replace("_", " ").title()+" Pct Diff From 8/17"}) #Dict of all existing col names matched to col names + Pct Diff from 8/17
    holdf.rename(columns=rename, inplace=True)
    df.rename(columns=rename, inplace=True)
    df=df.sort_values(by=["Date","cbsa_code"]).set_index(["Date","cbsa_code"])
    smoldf=holdf[list(rename.values())]
    df[list(rename.values())]=holdf[list(rename.values())].values #Overwrite
    return df
    
def prepcombos (df, splitcol, minlag, maxlag): #filepath to the data, first column in the lagged section, minimum lag, maximum lag
    data= df
    #Generate 2 lists, 1 of GWich and 1 of REstate
    split=list(data.columns).index(splitcol) 
    gcols=list(data.iloc[:,:split].columns)
    rcols=list(data.iloc[:,split:].columns)
    rcols.remove("Cbsa Title")
    rcols.remove("Householdrank")
    cbsas=list(data.index.get_level_values(1).unique().sort_values()) #Grab all CBSAS
    lags=list(np.arange(minlag, maxlag))
    combos=list(product(lags, cbsas, rcols)) #Mix 'em up
    return combos


def lagger(df, split, lagmonths, ref):
    df=df.iloc[:, :split] #Greenwich Data from joined data
    estate=ref.iloc[:-1].reset_index() #the Realtor.com data has a last line that's got a comment in 1 box and no values in the others; it throws off joins/calcs so we need to ditch it
    real=estate.rename(columns = {"cbsa_code":"CBSA"}) #Make relevant columns more human-readable
    real["Date"]=pd.to_datetime(real["Date"]) #Changes date fromat from YYYYmm to YYYY-mm-01 and renders it a string so we can join without needing datetime objects
    real["CBSA"]=real["CBSA"].astype("int64") #CBSA always comes out of prettymaster as int64, it was easier to typechange here instead of on prettymaster so we can join (cannot join objects to ints)
    real["Date"]=real["Date"]+pd.DateOffset(months=-lagmonths) #Offset the real estate data by X months so each join will see the Greenwich data joined to real estate data lagmonths later
    newname={}
    for i in real.columns:
        newname.update({i:i.replace("_", " ").title()}) 
    real=real.rename(columns=newname)
    real=real.rename(columns={"Cbsa":"CBSA"})
    real=real.set_index(["Date", "CBSA"]) #Multiindex so they line up
    real["Real Estate Lag Period (Months)"]=lagmonths
    #real.drop(columns="Quality Flag", inplace=True)
    return df.join(real)

def automated_regression(data, combos, lagref):
    today = str(datetime.date.today())
    count=0
    hold=[]
    for l, c, t in combos: #lag, cbsa, target
        lagged=lagger(data, 6, l, lagref)
        dd=lagged.swaplevel().loc[c] #Grab a dataframe consisting of just one CBSA
        dd=dd.dropna() #get rid of na's, we can't work our magic with them around
        if len(dd)<10: #check that there's enough data to actually split into the necessary number of folds
            continue #This is a keyword that skips the rest of the loop and dumps you back at the top
        exp=setup(data=dd, target=t, silent=True, fold=2) #Set it up
        modeltype=get_model_name((compare_models(sort="R2", include=["lr", "br", "omp", "dt", "par", "en", "llar", "svm"]))) #Store the shorthand
        if pull()["R2"][0]>=0.8: #If we're over threshold, just grab it and save it
            best=create_model(modeltype)
            final=finalize_model(best)
            if str(t)[-1]=="7": save_model(final, today+ " " + modeltype + " ("+ str(l) +"lag-"+str(c)+"-"+str(t)[:-19]+ ") plain R2-"+str(pull()["R2"][0])+" "+str(count))
            else: save_model(final, today+ " " + modeltype + " ("+ str(l) +"lag-"+str(c)+"-"+str(t)+ ") plain " +str(count))
            count=count+1
        elif pull()["R2"][0]<0.8 and pull()["R2"][0]>=0.6: #If we're near threshold, try tuning
            best=create_model(modeltype)
            b4=pull().loc["Mean"]["R2"]
            tuned=tune_model(best, optimize="R2", fold=3, choose_better=True)
            aftr=pull().loc["Mean"]["R2"]
            if aftr > 0.8: #If we break threshold, great
                final=finalize_model(tuned)
                if str(t)[-1]=="7": save_model(final, today+ " " + modeltype + " ("+ str(l) +"lag-"+str(c)+"-"+str(t)[:-19]+ ") tuned R2-"+str(pull()["R2"][0])+" "+str(count))
                else: save_model(final, today+ " " + modeltype + " ("+ str(l) +"lag-"+str(c)+"-"+str(t)+ ") tuned " +str(count))
                count=count+1
            elif aftr >= b4 and aftr <0.8: #If we don't, but we're still better than where we started, let's try bagging
                bagged=ensemble_model(best)
                if pull().loc["Mean"]["R2"] > 0.8:
                    final=finalize_model(bagged)
                    if str(t)[-1]=="7": save_model(final, today+ " " + modeltype + " ("+ str(l) +"lag-"+str(c)+"-"+str(t)[:-19]+ ") bagged R2-"+str(pull()["R2"][0])+" "+str(count))
                    else: save_model(final, today+ " " + modeltype + " ("+ str(l) +"lag-"+str(c)+"-"+str(t)+ ") bagged " +str(count))
                    count=count+1
                else: pass
            else: pass
        else: pass
        
#Read in and prepare anchor values to normalize outputs
cwd=os.getcwd().replace("\\", "/")
refd=cwd + "RDC_Inventory_Core_Metrics_Metro_History.csv",
real=pd.read_csv(refd, dtype="object")
real.drop("quality_flag", axis=1, inplace=True)
mid=real.iloc[:-1] #Drop the last row, which contains a text note
mid=mid.rename(columns={"month_date_yyyymm":"Date"}) #Make it more human-readable
mid["Date"]=pd.to_datetime(mid["Date"], format="%Y%m").astype(str)
lagref=anchordate(mid, "2017-08-01") #Choose an "anchor date" to refer to
from pycaret.regression import *
splitcol="Cbsa Title" #Choose the first Real Estate data Column to split our data in 2 to be lagged on

trfp=cwd + "/joined data.csv" #Read in our joined jobs data
raw=pd.read_csv(trfp)
raw.drop("Quality Flag Pct Diff From 8/17", axis=1, inplace=True)
raw.drop('Price Increased Count Pct Diff From 8/17', axis=1, inplace=True)
raw=raw.loc[raw["Householdrank"]<=20] #Pick the top 20 CBSAs to keep the load low; it's proof-of-concept
raw=raw.set_index(["Date", "CBSA"])
split=list(raw.columns).index(splitcol)

os.chdir("Saved Models") #Move to a folder to save our significant models

combos= prepcombos(raw, splitcol, 4, 13) #Prepare combinations of lags-cbsas-targets for automated checking
try: 
    automated_regression(raw, combos, lagref)
except KeyError as e:
    print("KeyError")
    print(e)
    pass
except ValueError as e: 
    print(e)
    pass