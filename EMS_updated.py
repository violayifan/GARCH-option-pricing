#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from functools import partial
import scipy
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
from statsmodels.base.model import GenericLikelihoodModel
from arch import arch_model
from statsmodels.tools.numdiff import approx_hess,approx_fprime
from arch.univariate import ConstantMean
import itertools
import datetime
from scipy.stats import norm
from scipy.optimize import brentq

import logging


logger = logging.getLogger('EMS_fx')  
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('EMS_fx.log')  
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  
fh.setFormatter(formatter)  
ch.setFormatter(formatter)  
logger.addHandler(fh)  
logger.addHandler(ch)  



class PayoffType(Enum):
    Call=0
    Put=1
    
class EuropeanOption():
    def __init__(self,expiry,strike,payoffType):
        self.expiry = expiry
        self.strike = strike
        self.payoffType = payoffType
    def payoff(self,S):
        if self.payoffType == PayoffType.Call:
            return max(S-self.strike,0)
        elif self.payoffType == PayoffType.Put:
            return max(self.strike-S,0)
        else:
            raise Exception('payoffType not supported:',self.payoffType)
            
    def valueAtNode(self,t,S,continuation):
        return continuation
        

def garch_ems(sigma0,dw,beta0,beta1,beta2,gamma1,volume,lbd):
    return beta0 + beta1 * sigma0 + beta2 * sigma0 * (dw - lbd)**2 + gamma1 * volume


def ems(S0,sigma0_2,r,mIntervals,nPaths,trade,beta0,beta1,beta2,gamma1,volume,lbd):
    "Empirical Martingale simulation for option pricing."
    
    value = 0
    value1 = 0
    hsquare = 0
    hsquare1 = 0
    sigma_2 = sigma0_2*np.ones((nPaths))
    dw = np.random.normal(0,1,nPaths)
    Sbar = np.zeros((mIntervals,nPaths))
    Sbar[0] = S0 * np.ones((nPaths))
    Sast = np.zeros((mIntervals,nPaths))
    Sast[0] = Sbar[0]
    sigmalist = np.zeros((mIntervals,nPaths))
    sigmalist[0] = sigma_2
    for m in range(1,mIntervals):
        sigma_2 = garch_ems(sigma_2,dw,beta0,beta1,beta2,gamma1,volume[m-1],lbd)
        sigmalist[m] = sigma_2
        dw = np.random.normal(0,1,nPaths)
        S = Sbar[m-1] * np.exp(r - 0.5 * sigma_2 + np.sqrt(sigma_2) * dw)
        Sbar[m] = S
        Z = Sbar[m] / Sbar[m-1] * Sast[m-1]
        Z0 = np.exp(-m * r) * sum(Z) / nPaths
        Sast[m] = S0 * Z/ Z0
    for s in Sast[-1]:
        h = trade.payoff(s)
        value += h
        hsquare += h**2
    for s in Sbar[-1]:
        h = trade.payoff(s)
        value1 += h
        hsquare1 += h**2
        
    pv1 = np.exp(-r * mIntervals) * value1 / nPaths    
    pv = np.exp(-r * mIntervals) * value / nPaths
    stderr = np.sqrt((hsquare / nPaths - (value / nPaths)**2) / nPaths)
    stderr1 = np.sqrt((hsquare1 / nPaths - (value1 / nPaths)**2) / nPaths)
    return Sbar, Sast, pv, pv1, stderr, stderr1, sigmalist


#Under P measure, the stock process ln(S(t+1)/S(t))=mu+sigma_t+1*eplison_t+1, GARCH process is
#sigma(t+1)^2=beta0+beta1*sigma(t)^2+beta2*(sigma_t*eplison_t)^2+sum(gamma(t)*X(t)).
#stock log return is observable

def backcast(resids) -> float:
    " used to generate inital sigma value for GARCH recursion,\
    it is the weighted average of maximum 75 approximated residuals \
    with a expotential smoothing factor 0.94."
    
    power = 2
    tau = min(75, resids.shape[0])
    w = 0.94 ** np.arange(tau)
    w = w / sum(w)
    backcast = np.sum((abs(resids[:tau]) ** power) * w)

    return float(backcast)
    
def ewma_recursion(
    lam: float, resids, sigma2, nobs: int, backcast: float):
    
    "Exponentially Weighted Moving-Average Variance process."

    # Throw away bounds
    var_bounds = np.ones((nobs, 2)) * np.array([-1.0, 1.7e308])
    
    parameters = np.array([0.0, 1.0 - lam, lam])
    
    p, q= 1, 1
    fresids = resids **2

    for t in range(nobs):
        loc = 0
        sigma2[t] = parameters[loc]
        loc += 1
        for j in range(p):
            if (t - 1 - j) < 0:
                sigma2[t] += parameters[loc] * backcast
            else:
                sigma2[t] += parameters[loc] * fresids[t - 1 - j]
            loc += 1
        for j in range(q):
            if (t - 1 - j) < 0:
                sigma2[t] += parameters[loc] * backcast
            else:
                sigma2[t] += parameters[loc] * sigma2[t - 1 - j]
            loc += 1
        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])

    return sigma2


def variance_bounds(logS,theta):
    "Create variance bounds to cap squared sigma,ensure optimizer doesn't \
    produce invalid values, use EWMA to smooth out the bounds periodically."
    
    resids = logS - theta[3]
    nobs = resids.shape[0]
    tau = min(75, nobs)
    w = 0.94 ** np.arange(tau)
    w = w / sum(w)
    var_bound = np.zeros(nobs)
    initial_value = w.dot(resids[:tau] ** 2.0)
    ewma_recursion(0.94, resids, var_bound, resids.shape[0], initial_value)
    
    var_bounds = np.vstack((var_bound / 1e6, var_bound * 1e6)).T
    var = resids.var()
    min_upper_bound = 1 + (resids ** 2.0).max()
    lower_bound, upper_bound = var / 1e8, 1e7 * (1 + (resids ** 2.0).max())
    var_bounds[var_bounds[:, 0] < lower_bound, 0] = lower_bound
    var_bounds[var_bounds[:, 1] < min_upper_bound, 1] = min_upper_bound
    var_bounds[var_bounds[:, 1] > upper_bound, 1] = upper_bound
    
    return np.ascontiguousarray(var_bounds)

def bounds_check(sigma2: float, var_bounds):
    if sigma2 < var_bounds[0]:
        sigma2 = var_bounds[0]
    elif sigma2 > var_bounds[1]:
        if not np.isinf(sigma2):
            sigma2 = var_bounds[1] + np.log(sigma2 / var_bounds[1])
        else:
            sigma2 = var_bounds[1] + 1000
    return sigma2

def squared_sigmas_Pmeasure(logS,volume,theta,var_bounds,backcast):
    "Use empirical data to formalize recursive GARCH process with exogenous\
    variables."
    
    beta0 = theta[0]
    beta1 = theta[1]
    beta2 = theta[2]
    mu = theta[3]
    lbd = theta[4]
    gamma1 = theta[5]
    
    var_bounds = var_bounds(logS,theta)
    backcast = backcast(logS-mu)
    n = len(logS)
    sigma2 = np.ndarray(n)

    for t in range(n):
        if (t -1) < 0:
            sigma2[t] = beta0 + beta1 * backcast + beta2 * backcast
        else:
            sigma2[t] = beta0 + beta1 * sigma2[t-1] + beta2 * sigma2[t-1] *\
    ((logS[t-1] - mu) / np.sqrt(sigma2[t-1]) - lbd)**2 + gamma1 * volume[t]
        
        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])
        
    return sigma2
    
def log_likelihood(logS,volume,var_bounds,backcast,theta, individual=False):
    n = len(logS) 
    mu = theta[3]
    sigma_sqd = squared_sigmas_Pmeasure(logS,volume,theta,var_bounds,backcast)
    fun = [-np.log(np.sqrt(2 * np.pi))-((logS[t]-mu) ** 2)/(2 * sigma_sqd[t])-\
         0.5 * np.log(sigma_sqd[t]) for t in range(n)]
    if individual:
        return np.array(fun)
    else:
        return -sum(fun)

def starting_values(logS,volume,ngamma,var_bounds,backcast):
    "To pick up close-optimal inital guesses by creating certain combinations of \
    random parameters to input into log-likelihood funcionts. The pair with \
    highest likelihhod value will be the good starting values for optimizer."
    
    p, q = 1,1
    mu,lbd = 1,1
    power = 2
    alphas = np.linspace(0.01, 0.2,3)
    alphas1= alphas
    alphas2= alphas
    betas = alphas
    zetas = [alphas for i in range(ngamma)]
    abg = [0.5, 0.7, 0.9, 0.98]
    abgs = list(itertools.product(*([alphas, abg, betas, alphas1, alphas2]+(zetas))))
        
    resids=logS-np.mean(logS)
    target = np.mean(abs(resids) ** power)
    svs = []
    llfs = np.zeros(len(abgs))
    for i, values in enumerate(abgs):
        alpha, agb, *zeta = values
        sv = (1.0 - agb) * target * np.ones(p + mu + lbd + q + 1+ngamma)
        if q > 0:
            sv[1 : 1 + p] = agb
        if mu > 0:
            sv[3 : 3 + mu] = alpha
        if lbd >0:
            sv[4 : 4 + lbd] = alpha
        if p > 0:
            sv[1 + p  : 1 + p + q] = alpha
        svs.append(sv)
        sigma2=squared_sigmas_Pmeasure(logS,volume,sv,var_bounds,backcast)
        resids=logS-sv[3]
        lls=-0.5 * (np.log(2 * np.pi) + np.log(sigma2) + resids ** 2.0 / sigma2)
        llfs[i] =sum(lls)
    loc = np.argmax(llfs)

    return svs[int(loc)]

def params_estimate(logS,volume,var_bounds,backcast,sv,method,xmean):
    cons=(\
          {'type':'ineq','fun':lambda x:np.array([1-x[1]-x[2]*(1+x[4]**2)])},
           {'type':'ineq','fun':lambda x:np.array(x[1])},
           {'type':'ineq','fun':lambda x:np.array(x[2])},
           {'type':'ineq','fun':lambda x:x[0]+x[5]*xmean})
    
    resids = logS - np.mean(logS)
    v= np.mean(abs(resids) ** 2)
    
    objective = partial(log_likelihood,logS,volume,var_bounds,backcast,individual=False)
    
    bnds = ((0,10 * v),(0,1),(0,1),(0,1),(0,1),(None,None))

    result = scipy.optimize.minimize(objective,sv,
                                   method = method,
                                   bounds = bnds,
                                   constraints = cons)
    
    return result
def Sts_interference(func, result, names:list ) -> pd.DataFrame:
    
    " With the numercial estimation of Jacobian, use BHHH estimator to calcualte\
    covariance matrix, with the virtue of non-negative definite, so that standard\
    errors, T-statistic and P-values could all be calculated."
    
    params = result.x
    jac = approx_fprime(params,func)
    btt = jac.T.dot(jac)
    inv_btt = np.linalg.inv(btt)
    stderr = np.sqrt(np.diag(inv_btt))
    params_df = pd.Series(params, index = names, name='Coef')
    stderr_df = pd.Series(stderr, index = names, name="std_err")
    tvalues_df  = pd.Series(params / stderr, index=names, name='t')
    pvalues_df = pd.Series(stats.norm.sf(np.abs(params / stderr)) * 2, 
                        index = names, name="P>|t|")
    
    return pd.concat([params_df, stderr_df, tvalues_df, pvalues_df],axis=1)
    
    

if __name__=='__main__':
    
#    GARCH estimation & backtesting using market option price
    
    path='SPX Option/SPX stock prices/'
    year=[1996+i*1 for i in range(24)]
    stockp=[]
    volume=[]
    for y in year:
        stockp.append(pd.read_csv(path+'SPX'+' '+str(y)+' '+'Stock Prices.csv',header=0,index_col=0)['Close'])
        volume.append(pd.read_csv(path+'SPX'+' '+str(y)+' '+'Stock Prices.csv',header=0,index_col=0)['Volume'])
    np.random.seed(2)
    
    
    #get raw data
    
    #vol of return
    stockdf = pd.concat(stockp,axis=0)
    
    #macro factor
    volumedf = pd.concat(volume,axis=0)
    
    interest_rate=pd.read_excel("interest_rate.xlsx",parse_dates=True,header=0,index_col=0)
    treasury_yield=pd.read_csv("Quandl Zero Curve FED-SVENY.csv",parse_dates=True,header=0,index_col=0,usecols=["Date","SVENY10"])/100
    foreign_exchange=pd.read_excel("foreign_exchange.xlsx",parse_dates=True,header=0,index_col=0).dropna()
    
    gold_future=pd.read_excel("gold futures.xlsx",parse_dates=True,header=0,index_col=0,usecols=["Date","Price"])[::-1]
    oil_future=pd.read_excel("Crude Oil WTI Futures.xlsx",parse_dates=True,header=0,index_col=0,usecols=["date","price"])
    
    stock=pd.DataFrame(stockdf,index=pd.to_datetime(stockdf.index))
    vol=pd.DataFrame(volumedf,index=pd.to_datetime(volumedf.index))
    
    fx_vol=pd.read_excel("exchange_vol.xlsx",parse_dates=True,header=0,index_col=0,usecols=["date","exchange rate daily vol"])[::-1]
    
    dic_factor={"volume":vol,"federal":interest_rate,"treasury":treasury_yield,"fx":foreign_exchange,\
                "gold_future":gold_future,"oil_future":oil_future,"fx_vol":fx_vol}

    
    #interest rate
    df_rate=pd.read_csv("Quandl Zero Curve FED-SVENY.csv",parse_dates=True,header=0,index_col=0,usecols=["Date","SVENY01"])/100
    #option price
    df_market=pd.read_csv("market_data.csv",parse_dates=True,header=0,index_col=0)
      
    #data processing (get parameter calculation sample,return sample array)
    def get_sample(rtdf,factordic,factorname,startdate,window,pricing_date=""):
        
        factordf=factordic[factorname]
        rtdf["logstock"]=np.log(rtdf/rtdf.shift())[1:]
  
        start=pd.to_datetime(startdate)
        l_date=list(rtdf.index)
        i=l_date.index(start)
        threshold=len(l_date)-window
        if i>=threshold:

            end=pd.to_datetime(pricing_date)
            start=end+datetime.timedelta(-window)
        
        else:
            end=l_date[i+window]
            
        logstock=rtdf[~rtdf.isin([-np.inf,np.inf])]
        logstock=logstock[(rtdf.index>=start)&(rtdf.index<=end)]["logstock"].dropna()
        factor_sub=factordf[(factordf.index>=start)&(factordf.index<=end)]
        
        df_merge=pd.merge(logstock,factor_sub,left_index=True,right_index=True)
        
        if factorname=="volume":    
            logS=df_merge.iloc[:-1,0].values
            factor=np.log(df_merge.iloc[1:,1]).values
        else:
            logS=df_merge.iloc[:,0].values
            factor=df_merge.iloc[:,1].values
        return logS,factor,end
    
    '''
    Backtesting
    '''
    '''
    overall result
    '''
    
    df_parameter=pd.DataFrame()
    df_option_sample=pd.DataFrame()
    #report daily rms(total,call,put)
    df_pricing_rst=pd.DataFrame(columns=["g_put_d","g_put_nd","g_call_d","g_call_nd",\
                                                 "bs_put_d","bs_put_nd","bs_call_d","bs_call_nd"])
    
    #get sample for garch parameter estimation
    l_date=list(stockdf.index)
    threshold=len(l_date)-252*2
    count=0
    n=0
    macro_factor="fx"
    start_date=l_date.index("2007-01-03")
    for date in stockdf.index[start_date::20]:
        starttime=datetime.datetime.now()
        if l_date.index(date)<threshold:
            sample=get_sample(stock,dic_factor,macro_factor,date,window=252*2,pricing_date="")
           
        else:
            try:
                end_date_index=l_date.index(date)+20*count
                end_date=l_date[end_date_index]
                sample=get_sample(stock,dic_factor,macro_factor,date,window=252*2,pricing_date=end_date)
                count=count+1
            except:
                break
                    
        logS=sample[0]
        factor=sample[1]
        end_date=sample[2]
        xmean=np.mean(factor)
        #start estimate parameters in garch
        sv = starting_values(logS,factor,1,variance_bounds,backcast)
        status=0
        
        result=params_estimate(logS,factor,variance_bounds,backcast,sv,'Nelder-Mead',xmean)
        beta0,beta1,beta2,lbd,gamma=result.x[0],result.x[1],result.x[2],result.x[4],result.x[5]
                
        try:
            #get statistic test result 
            names = ['Beta0','Beta1', 'Beta2', 'Mu', 'Lambda', 'Gamma1']
            func = partial(log_likelihood,logS,factor,variance_bounds, backcast, individual=True)
            Model_results_df = Sts_interference(func,result, names)
            p_value=Model_results_df.iloc[:,-1]
            p_beta0,p_beta1,p_beta2,p_mu,p_lambda,p_gamma=p_value[0],p_value[1],p_value[2],\
                                                                      p_value[3],p_value[4],p_value[5]
        except: 
            continue                                                          
               
        if result.success==True and beta0>0 and beta1 >0 and beta2 >0 and lbd >0 and 1-beta1-beta2*(1+lbd**2)>0:
            status=1          
            finalresult=result
    
            print("get parameter")
                            
        else:
            print('Invalid estimation')
    
        
        if status==0:
            continue
                   
        #get final estimated parameter
        beta0,beta1,beta2,mu,lbd,gamma=finalresult.x[0],finalresult.x[1],finalresult.x[2],finalresult.x[3],finalresult.x[4],result.x[5]
        df_parameter=df_parameter.append([{"date":date,"beta0":beta0,"beta1":beta1,"beta2":beta2,\
                                          "mu":mu,"lbd":lbd,"gamma":gamma,\
                                        "p_beta0":p_beta0,"p_beta1":p_beta1,"p_beta2":p_beta2,"p_mu":p_mu,"p_lambda":p_lambda,"p_gamma":p_gamma}])
        
        sigma0_garch=abs((beta0+gamma*xmean)/(1-beta1-beta2*(1+lbd**2)))
        sigma_bs=beta0
        #start option pricing test after parameter estimation updated
        
        for date_p in pd.date_range(end_date,end_date+datetime.timedelta(19)):
            
            try:
                test=df_market.loc[end_date,:].sort_values(by="Strike")
            except:
                continue
            #daily pricing error list by category
            print("start option pricing")
            
            #risk free rate
            try:
                r=(1+df_rate.loc[date_p].values[0])**(1/360)-1
            except:
                r=r

            
            df_daily_error=pd.DataFrame(columns=["g_put_d","g_put_nd","g_call_d","g_call_nd",\
                                                 "bs_put_d","bs_put_nd","bs_call_d","bs_call_nd"])
           
            trade=EuropeanOption(90/360,1000,PayoffType.Put)
            S0=test["stock market price"][0]
            
            ems_garch=ems(S0,sigma0_garch,r,90,10000,trade,beta0,beta1,beta2,gamma,factor,lbd)
            state_price_garch=ems_garch[1]
            
            ems_bs=ems(S0,sigma_bs,r,90,10000,trade,beta0,0,0,0,factor,0)
            state_price_bs=ems_bs[1]
            
            #select out-the-money option
            for d,v in test.iterrows():
                
                strike,price=v["Strike"],v["Option Price"]
            
                if strike< S0:
                    if v["Type"]=="P":
                        est_garch_price=np.exp(-r * 90)*np.mean([max(i,0) for i in strike-state_price_garch[-1,:]])
                        est_bs_price=np.exp(-r * 90)*np.mean([max(i,0) for i in strike-state_price_bs[-1,:]])
                        
                    else:
                        continue
            
                if strike>=S0:
                    if v["Type"]=="C":
                        est_garch_price=np.exp(-r * 90)*np.mean([max(i,0) for i in state_price_garch[-1,:]-strike])
                        est_bs_price=np.exp(-r * 90)*np.mean([max(i,0) for i in state_price_bs[-1,:]-strike])
                    else:
                        continue
                    
                df_option_sample=df_option_sample.append([v])         
                error_garch=(est_garch_price-price)**2
                error_bs=(est_bs_price-price)**2
                moneyness=strike/S0
                
                if moneyness<0.85:
                    df_daily_error=df_daily_error.append(pd.DataFrame([{"g_put_d":error_garch,"bs_put_d":error_bs}]))
                if moneyness>=0.85 and moneyness<1:
                    df_daily_error=df_daily_error.append(pd.DataFrame([{"g_put_nd":error_garch,"bs_put_nd":error_bs}]))
                if moneyness>=1 and moneyness<1.15:
                    df_daily_error=df_daily_error.append(pd.DataFrame([{"g_call_nd":error_garch,"bs_call_nd":error_bs}]))
                if moneyness>=1.15:
                    df_daily_error=df_daily_error.append(pd.DataFrame([{"g_call_d":error_garch,"bs_call_d":error_bs}]))
            
            
            dic_error={}
            for col in df_daily_error.columns:
                rms=np.sqrt(np.mean(df_daily_error[col].dropna()))
                dic_error[col]=rms
            
            df_pricing_rst=df_pricing_rst.append(pd.DataFrame([dic_error],index=[date_p]))
            endtime=datetime.datetime.now()
            time=endtime-starttime
            n=n+1
            print(time.seconds,"parameter estimation date",date,"pricing date",date_p,"{}/{}".format(n,len(stockdf.index)))
            logger.info("parameter estimation date {} , pricing date{}".format(date,date_p))
 
    '''
    Statistic summary
    '''
    
    eco_factor="fx"
    
    
    #summary of parameter
    summary=df_parameter.set_index(["date"]).loc[:,"beta0":"gamma"]
    df_summary_parameter=pd.DataFrame([summary.mean(),summary.std()]).T
    
    df_pvalue=df_parameter.loc[:,"p_beta0":"p_gamma"]
    dic_pvalue={}
    for col in df_pvalue.columns:
        df_sub=df_pvalue[col].copy()
        ratio=len(df_sub[df_sub<0.05])/len(df_sub)
        dic_pvalue[col]=np.round(ratio,2)
        
    df_summary_parameter["significant ratio"]=list(dic_pvalue.values())
    df_summary_parameter=df_summary_parameter.rename(columns={0:"Mean",1:"SD"})
    df_summary_parameter.to_excel("parameter_summary_{}.xlsx".format(eco_factor))
    
    #overall result
    
    #group by option
    rms_mean_option=np.round(df_pricing_rst.mean().values,2)
    df_summary_option=pd.DataFrame({"Garch":list(rms_mean_option[[0,1,3,2]]),"BS":list(rms_mean_option[[4,5,7,6]])})
    df_summary_option.index.set_names(["Moneyness"],inplace=True)
    df_summary_option.rename(index={0:"K/S<0.85",1:"0.85≤K/S<1",2:"1≤K/S<1.15",3:"K/S≥1.15"},inplace=True)
    df_summary_option.to_excel("rms_option_{}.xlsx".format(eco_factor))


    #group by year
    df_summary_year=np.round(df_pricing_rst.resample("2A").mean()[1:],2)
    df_summary_year.fillna(df_summary_year.mean(),axis=0,inplace=True)
    
    garch_rms=df_summary_year.loc[:,["g_put_d","g_put_nd","g_call_nd","g_call_d"]].values.reshape(-1,1)
    bs_rms=df_summary_year.loc[:,["bs_put_d","bs_put_nd","bs_call_nd","bs_call_d"]].values.reshape(-1,1)
    
    index_year=["2009~2011"]*4+["2011~2013"]*4+["2013~2015"]*4+["2015~2017"]*4+["2017~2019"]*4
    index_moneyness=["K/S<0.85","0.85≤K/S<1","1≤K/S<1.15","K/S≥1.15"]*5
    
    df_summary_year_rst=pd.DataFrame({"Garch":garch_rms[:,0],"BS":bs_rms[:,0],"Year":index_year,"Moneyness":index_moneyness}).set_index(["Year","Moneyness"])
    
    df_summary_year_rst.to_excel("rms_year_{}.xlsx".format(eco_factor))
    
    
            
    """
    Test on a particular date
    """
            
    #get parameter on a given date
    para_date="2017-02-15"
    test_date="2019-02-20"
    

    
    beta0,beta1,beta2,mu,lbd,gamma=df_parameter.set_index("date").loc[para_date,"beta0":"gamma"].values
    
    #test market sample
    sample_test=df_market.loc[pd.to_datetime(test_date),:].sort_values(by="Strike")
    factor_test=get_sample(stock,dic_factor,eco_factor,pd.to_datetime(para_date),window=252*2,pricing_date="")[1]
    
    xmean_test=np.mean(factor_test)
    
    #risk free rate
    try:
        r=(1+df_rate.loc[pd.to_datetime(test_date)].values[0])**(1/360)-1
    except:
        r=r
    
    # test pricing error
    l_error=[]
    l_strike=[]
    l_price=[]
    l_est_price=[]
    sigma0_garch=abs((beta0+gamma*xmean_test)/(1-beta1-beta2*(1+lbd**2)))
    
# test implied vol skewness
    def BSModelCall(S,K,r,sigma,T):
        d1=(np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
        d2=d1-sigma*np.sqrt(T)
        Vc=S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
        return Vc
 
    def BSModelPut(S,K,r,sigma,T):
        d1=(np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
        d2=d1-sigma*np.sqrt(T)
        Vp=K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
        return Vp

    # calculate implied voliatility
    def BSImpliedVol(func,S,K,r,T,price):
        impliedvol=brentq(lambda x:func(S, K, r, x, T)-price,-1,1)
        return impliedvol


    l_vol_mkt=[]
    l_vol_model=[]
    
    trade=EuropeanOption(90/360,1000,PayoffType.Put)
    S0_test=sample_test["stock market price"][0]
            
    ems_garch=ems(S0,sigma0_garch,r,90,10000,trade,beta0,beta1,beta2,gamma,factor_test,lbd)
    state_price_garch=ems_garch[1]
            
    #get risk free rate & select out-the-money option
    for d,v in sample_test.iterrows():
                
        strike,price=v["Strike"],v["Option Price"]
            
        if strike< S0:
            if v["Type"]=="P":
                est_garch_price=np.exp(-r * 90)*np.mean([max(i,0) for i in strike-state_price_garch[-1,:]])
                vol_mkt=BSImpliedVol(BSModelPut,S0,strike,r,90/360,price)
                vol_model=BSImpliedVol(BSModelPut,S0,strike,r,90/360,est_garch_price)
       
            else:
                continue
            
        if strike>=S0:
            if v["Type"]=="C":
                est_garch_price=np.exp(-r * 90)*np.mean([max(i,0) for i in state_price_garch[-1,:]-strike])
                vol_mkt=BSImpliedVol(BSModelCall,S0,strike,r,90/360,price)
                vol_model=BSImpliedVol(BSModelCall,S0,strike,r,90/360,est_garch_price)

            else:
                continue
                    
        error_garch=abs((est_garch_price-price)/price)
        l_error.append(error_garch)
        l_strike.append(strike)
        l_price.append(price)
        l_est_price.append(est_garch_price)
        l_vol_mkt.append(vol_mkt)
        l_vol_model.append(vol_model)
        
    
    l_strike=list(np.array(l_strike)/S0_test)
    fig, ax = plt.subplots()
    plt.plot(l_strike,l_price,label="market price")
    plt.plot(l_strike,l_est_price,label="estimated price")
    plt.xlabel("Moneyness")
    plt.ylabel("Option Price")
    plt.title("Price (Market vs Model)")
    plt.legend()
    plt.savefig("Price_{}.png".format(eco_factor))
    
    fig, ax = plt.subplots()
    plt.plot(l_strike,l_vol_mkt,label="market price")
    plt.plot(l_strike,l_vol_model,label="estimated price")
    plt.xlabel("Moneyness")
    plt.ylabel("Implied vol")
    plt.title("Implied vol (Market vs Model)")
    plt.legend()
    plt.savefig("Vol_{}.png".format(eco_factor))

    
    fig,ax=plt.subplots()
    plt.plot(l_strike,l_error,label="garch "+"FX")
    plt.xlabel("Moneyness")
    plt.ylabel("Absolute Error %")
    plt.title("Pricing error")
    plt.legend()
    plt.savefig("Error_{}.png".format("FX"))


'''
stock price distribution
'''
plt.hist(state_price_bs[-1],density=True)
plt.hist(state_price_garch[-1],density=True)
plt.title("State price at maturity predicted with treasury")


#     beta0,beta1,beta2,mu,gamma1=0.00001,0.7,0.2,0,0
#     model=arch_model(logstock[1:700].values).fit()
#     print(model.summary())
    
    
# # Option pricing    
    
    
#     gamma1=0
#     sigma0_21=abs(beta0/(1-beta1-beta2))
# #   sigma0_2=(logstock[2]/np.random.normal(0,1))**2
#     S0=stockdf.iloc[2]
#     r=(1+0.1)**(1/360)-1
# #    lbd=0.01
#     lbd2=np.mean((logstock[1:]-mu)/np.std(logstock[1:]))
#     trade=EuropeanOption(0.25,1100,PayoffType.Call)
#     volume=np.log(volumedf[1:253].values)
#     volume=factor[1:253]
#     Sbar,Sast,callems,callcrude,stderr_ems,stderr_crude,slist=ems(1300,sigma0_21,r,90,10000,trade,beta0,beta1,beta2,gamma,volume,lbd)
    
#     np.sqrt((np.mean(np.sum(slist,axis=0)))*12/9)
    
#     ems(1300,sigma0_21,r,90,10000,trade,beta0,beta1,beta2,gamma,volume,lbd)[2]
#     ems(1300,sigma0_21,r,90,10000,trade,beta0,beta1,beta2,gamma,volume,lbd)[2]

    
    
# # ploting and test on normality    
#     testy=(logstock[1:]-mu)/np.std(logstock[1:])
#     plt.hist(logstock[1:].values,bins=30)
#     plt.xlabel('Sigma')
#     plt.ylabel('Observation')
#     plt.title('log-returns sigma distribution')
#     plt.savefig('sigma distribution.jpg')
#     jarque_bera(logstock[1:].values)
    
#     y2 = np.random.normal(0, 1, 3514)
#     both = np.matrix([testy, y2])
#     plt.plot(both.T, alpha=.7);
#     plt.axhline(y2.std(), color='yellow', linestyle='--')
#     plt.axhline(-y2.std(), color='yellow', linestyle='--')
#     plt.axhline(3*y2.std(), color='red', linestyle='--')
#     plt.axhline(-3*y2.std(), color='red', linestyle='--')
#     plt.xlabel('time')
#     plt.ylabel('Sigma')
#     plt.title('Log-returns vs normal distribution')
#     plt.savefig('compare.jpg')