"""

George Goginashvil

NYCDSA

Project 4

Investment Analysis

"""

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import copy
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

"""

Part 1

Value Investing

"""


"""

Step 1

Screen stocks from Dow Jones with highest sharpe ratios


"""


tickers = ['AXP','AMGN','AAPL','BA','CAT','CSCO','CVX', 'GS','HD', 'HON', 
           'IBM','INTC','JNJ', 'KO', 'JPM','MCD', 'MMM', 'MRK', 'MSFT', 
           'NKE', 'PG', 'TRV','UNH', 'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS']

sp500_df = pd.DataFrame() 
dj_df = pd.DataFrame() 

data_dict = {}            
start = dt.datetime.today()-dt.timedelta(1825)
end = dt.datetime.today()


for ticker in tickers:
    data_dict[ticker] = yf.download(ticker,start,end)
    data_dict[ticker].dropna(inplace=True,how="all")
    
sp500_df = yf.download('^GSPC',start,end)
dj_df = yf.download('^DJI',start,end)


work_dict = copy.deepcopy(data_dict)
work_df = pd.DataFrame()

for ticker in tickers:
    work_df[ticker] = work_dict[ticker]["Adj Close"]
    
    
def annual_income(each_col):
    col_pct_change = each_col.pct_change(1)
    col_cum_pct_change = (1 + col_pct_change).cumprod()
    n = len(each_col) / 252
    a_i = (col_cum_pct_change[-1])**(1/n) - 1
    return a_i




def volatility(each_col):
    col_pct_change = each_col.pct_change(1)
    vol = col_pct_change.std() * np.sqrt(252)
    return vol



def sharpe(each_col, rf):
    sr = (annual_income(each_col) - rf) / volatility(each_col)
    return sr


def max_draw_down(each_col):
    col_pct_change = each_col.pct_change()
    col_cum_pct_change = (1 + col_pct_change).cumprod()
    col_cum_roll_pct_change = col_cum_pct_change.cummax()
    drawdown = col_cum_roll_pct_change - col_cum_pct_change
    drawdown_percentage = drawdown / col_cum_roll_pct_change 
    max_draw_down_percent = drawdown_percentage.max()
    return max_draw_down_percent


def calmar(each_col):
    a_i = annual_income(each_col)
    max_draw_down_percent = max_draw_down(each_col)
    calmar_ratio = a_i / max_draw_down_percent
    return calmar_ratio


annual_income_val = []
annual_volatility_val = []
sharpe_val = []
max_draw_down_val = []
calmar_val = []
risk_free = 0.0159

for ticker in tickers:
     annual_income_val.append(annual_income(work_df[ticker]))
     annual_volatility_val.append(volatility(work_df[ticker]))
     sharpe_val.append(sharpe(work_df[ticker], risk_free))
     max_draw_down_val.append(max_draw_down(work_df[ticker]))
     calmar_val.append(calmar(work_df[ticker]))

data_numbers_dow = {'Income': annual_income_val,
    'volatility': annual_volatility_val,
    'Sharpe': sharpe_val,
    'Drawdown': max_draw_down_val,
    'Calmar': calmar_val}


df_dow_numbers = pd.DataFrame(data_numbers_dow, index = tickers)

df_dow_numbers.plot.bar()


"""
Step 2

Screened Portfolio Descripitive Data

"""

tickers_portfolio = ['AAPL', 'CAT', 'CSCO', 'HD', 'JPM', 'MSFT', 'UNH', 'CRM', 'V', 'WMT']
portfolio_df = work_df[tickers_portfolio]

portfolio_df.plot(title='Adjusted Close', figsize=(16, 8))

portfolio_df_daily = pd.DataFrame()

for ticker in tickers_portfolio:
    portfolio_df_daily[ticker] = portfolio_df[ticker].pct_change(1)

scatter_matrix(portfolio_df_daily, figsize=(20, 20), alpha=0.2, hist_kwds={'bins': 100});

portfolio_df_daily['AAPL'].hist(bins=50, label='Apple', figsize=(10,8), alpha=0.4)
portfolio_df_daily['CAT'].hist(bins=50, label='CAT', figsize=(10,8), alpha=0.4)
portfolio_df_daily['CSCO'].hist(bins=50, label='Cisco', figsize=(10,8), alpha=0.4)
portfolio_df_daily['HD'].hist(bins=50, label='HomeDepot', figsize=(10,8), alpha=0.4)
portfolio_df_daily['JPM'].hist(bins=50, label='JPMorgan', figsize=(10,8), alpha=0.4)
portfolio_df_daily['MSFT'].hist(bins=50, label='Microsoft', figsize=(10,8), alpha=0.4)
portfolio_df_daily['UNH'].hist(bins=50, label='UnitedHealth', figsize=(10,8), alpha=0.4)
portfolio_df_daily['CRM'].hist(bins=50, label='Salesforce', figsize=(10,8), alpha=0.4)
portfolio_df_daily['V'].hist(bins=50, label='Visa', figsize=(10,8), alpha=0.4)
portfolio_df_daily['WMT'].hist(bins=50, label='Walmart ', figsize=(10,8), alpha=0.4)
plt.legend()



portfolio_df_daily_cum = pd.DataFrame()

for ticker in tickers_portfolio:
    portfolio_df_daily_cum[ticker] = (1 + portfolio_df_daily[ticker]).cumprod()
  
  
portfolio_df_daily_cum.plot(title='Cummulative Return', figsize=(16, 8)) 
  


"""

Step 3

Markowitz's Efficient Frontier


"""

portfolio_df_daily_log = np.log(portfolio_df / portfolio_df.shift(1))
portfolio_df_daily_log.head()

portfolio_df_daily_log.hist(bins=100, figsize=(20, 10))
plt.tight_layout()


num_run = 50000
stock_weights = np.zeros((num_run, len(portfolio_df.columns)))
stock_returns = np.zeros(num_run)
stock_volatilites = np.zeros(num_run)
stock_sharpes = np.zeros(num_run)


for ind in range(num_run):
    
    # Weights
    weights = np.array(np.random.random(10))
    weights = weights / weights.sum()
    
    # Save Weights
    stock_weights[ind, : ] = weights

    # Expected Return
    stock_returns[ind] = np.sum(portfolio_df_daily_log.mean() * weights * 252)

    # Expected Volatility
    stock_volatilites[ind] = np.sqrt(np.dot(weights.T, np.dot(portfolio_df_daily_log.cov() * 252, weights)))

    # Sharpe Ratio
    stock_sharpes[ind] = stock_returns[ind] / stock_volatilites[ind]


stock_sharpes.max()

"""

Max Sharpe Ratio 1.3156985310026847


"""

stock_weights[stock_sharpes.argmax()]

"""

Portfolio 22177 achieved highest Sharpe Ratio 

with stock weights

AAPL  0.235819487
CAT  0.238781615
CSCO  0.0106728
HD  0.038288569
JPM  0.004921114
MSFT  0.172376182
UNH  0.0692350
CRM  0.000139592
V  0.019577205
WMT  0.210188474

"""





plt.figure(figsize=(20, 10))
plt.scatter(stock_volatilites, stock_returns, c=stock_sharpes, cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

plt.scatter(stock_volatilites[stock_sharpes.argmax()], stock_returns[stock_sharpes.argmax()], c='red', s=50, edgecolors='black')
            
from scipy.optimize import minimize            


def ret_vol_sr(weights):
        weights = np.array(weights)
        ret = np.sum(portfolio_df_daily_log.mean() * weights * 252)
        vol = np.sqrt(np.dot(weights.T, np.dot(portfolio_df_daily_log.cov() * 252, weights)))
        sr = ret / vol
        return np.array([ret, vol, sr])     



def neg_sharpe(weights):
    return ret_vol_sr(weights)[2] * (-1)


def check_sum(weights):
    return np.sum(weights) - 1

cons = ({'type': 'eq', 'fun': check_sum})

bounds = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1))

init_guess = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

opt_results = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)

opt_results

ret_vol_sr(opt_results.x)

frontier_y = np.linspace(0, 0.4, 100)


def min_volatility(weights):
    return ret_vol_sr(weights)[1]


frontier_volatility = []

for i in frontier_y:
    cons = ({'type': 'eq', 'fun': check_sum}, {'type': 'eq', 'fun': lambda w : ret_vol_sr(w)[0] - i})
    result = minimize(min_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    frontier_volatility.append(result['fun'])


plt.figure(figsize=(20, 10))
plt.scatter(stock_volatilites, stock_returns, c=stock_sharpes, cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.title('Markowitz\'s Efficient Frontier')
plt.xlabel('Volatility')
plt.ylabel('Return')

plt.scatter(stock_volatilites[stock_sharpes.argmax()], stock_returns[stock_sharpes.argmax()], c='red', s=100, edgecolors='black')

plt.plot(frontier_volatility, frontier_y, 'g--', linewidth=3)




"""

Step 4

CAPM - Capital Asset Pricing Model


"""

capm_portfolio = portfolio_df
capm_portfolio['GSPC'] = sp500_df['Adj Close']
capm_portfolio['DJI'] = dj_df['Adj Close']


capm_portfolio_daily_log = np.log(capm_portfolio / capm_portfolio.shift(1))
capm_portfolio_daily_log.head()
capm_portfolio_daily_log.dropna(inplace=True,how="all")
capm_sp500_betas = []
capm_dj_betas = []
capm_annual_income = []


from scipy import stats

for ticker in tickers_portfolio:
    capm_sp500_betas.append(stats.linregress(capm_portfolio_daily_log[ticker], capm_portfolio_daily_log['GSPC'])[0])
    capm_dj_betas.append(stats.linregress(capm_portfolio_daily_log[ticker], capm_portfolio_daily_log['DJI'])[0])
    
for ticker in tickers_portfolio:
     capm_annual_income.append(annual_income(capm_portfolio[ticker]))


capm_sp500_annual_income = annual_income(capm_portfolio['GSPC'])
capm_dj_annual_income = annual_income(capm_portfolio['DJI'])
     

def CAPM(risk_free_rate, beta, market):
    expected_return = risk_free_rate + beta * (market - risk_free_rate)
    return expected_return
    

capm_expected_returns_sp500 = []
capm_expected_returns_dj = []

for i in range(len(tickers_portfolio)):
     capm_expected_returns_sp500.append(CAPM(risk_free, capm_sp500_betas[i], capm_sp500_annual_income))
     capm_expected_returns_dj.append(CAPM(risk_free, capm_dj_betas[i], capm_dj_annual_income))


capm_dict = {'Annual Income Actual': capm_annual_income,
    'Annual Income Expected SP500': capm_expected_returns_sp500,
    'Annual Income Expected DJ': capm_expected_returns_dj}


capm_returns_df = pd.DataFrame(capm_dict, index = tickers_portfolio)

capm_returns_df.plot.bar()


"""

Part 2

Algorithmic Trading 

Indicators


"""


"""

1. Moving Average Convergence Divergence

"""

portfolio_df.columns

macd_dict = {}


def MACD(col, a, b, c):
    df = pd.DataFrame()
    df['Adj Close'] = col
    df[a] = col.ewm(span=a, min_periods=a).mean()
    df[b] = col.ewm(span=b, min_periods=b).mean()
    df['MACD'] = df[a] - df[b]
    df['Signal'] = df['MACD'].ewm(span=c, min_periods=c).mean()
    df.dropna(inplace=True)
    return df


for ticker in tickers_portfolio:
    macd_dict[ticker] = MACD(portfolio_df[ticker], 12, 26, 9)


for ticker in tickers_portfolio:
    plt.subplot(2, 1, 1)
    plt.plot(macd_dict[ticker].iloc[-150 :, 0])
    plt.title(ticker + ' Moving Average Convergence Divergence')
    
    plt.subplot(2, 1, 2)
    plt.plot(macd_dict[ticker].iloc[-150 :, [3, 4]])
    plt.show()



"""

2. Bollinger Bands

"""

bollinger_dict = {}

def BLNBAND(col, n):
    df = pd.DataFrame()
    df['Adj Close'] = col
    df['MA'] = col.rolling(n).mean()
    df['UP'] = df['MA'] + 2 * col.rolling(n).std()
    df['DN'] = df['MA'] - 2 * col.rolling(n).std()
    df.dropna(inplace=True)
    return df


for ticker in tickers_portfolio:
    bollinger_dict[ticker] = BLNBAND(portfolio_df[ticker], 20)


for ticker in tickers_portfolio:   
    plt.subplot(2, 1, 1)
    plt.plot(bollinger_dict[ticker].iloc[-100 :, 0])
    plt.title(ticker + ' Bollinger Bands')
    
    plt.subplot(2, 1, 2)
    plt.plot(bollinger_dict[ticker].iloc[-100 :, [1, 2, 3]])
    plt.show()
    


"""

3. RSI - Relative Strength Index

"""

rsi_dict = {}

def RSI(col, n):
    df = pd.DataFrame()
    df['Adj Close'] = col
    df['Delta'] = col - col.shift(1)
    df['Gain'] = np.where(df['Delta'] >= 0, df['Delta'], 0)
    df['Loss'] = np.where(df['Delta'] <= 0, abs(df['Delta']), 0)
    avg_gain = []
    avg_loss = []
    gain = df['Gain'].tolist()
    loss = df['Loss'].tolist()
    
    for i in range(len(df)):
        if i < n:
            avg_gain.append(np.NAN)
            avg_loss.append(np.NAN)
        elif i == n:
            avg_gain.append(df['Gain'].rolling(n).mean().tolist()[n])
            avg_loss.append(df['Loss'].rolling(n).mean().tolist()[n])
        elif i > n:
            avg_gain.append(((n - 1) * avg_gain[i - 1] + gain[i]) / n)
            avg_loss.append(((n - 1) * avg_loss[i - 1] + loss[i]) / n)
    df['Avg_Gain'] = np.array(avg_gain)
    df['Avg_Loss'] = np.array(avg_loss)
    df['RS'] = df['Avg_Gain'] / df['Avg_Loss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))
    return df


for ticker in tickers_portfolio:
    rsi_dict[ticker] = RSI(portfolio_df[ticker], 14)


for ticker in tickers_portfolio:   
    plt.subplot(2, 1, 1)
    plt.plot(rsi_dict[ticker].iloc[-100 :, 0])
    plt.title(ticker + ' Relative Strength Index')
    
    plt.subplot(2, 1, 2)
    plt.plot(rsi_dict[ticker].iloc[-100 :, 7])
    plt.yticks(np.arange(0, 100, 20))
    plt.show()     
    
    
    
"""

4. ADX - Average Directional Index

"""  


adx_dict = {}

for ticker in tickers_portfolio:
    adx_dict[ticker] = work_dict[ticker]
    
def ATR(DF, n):
    df = DF.copy()
    df['H-l'] = abs(df['High'] - df['Low'])
    df['H-PC'] = abs(df['High'] - df['Adj Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Adj Close'].shift(1))
    df['TR'] = df[['H-l', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    df1 = df.drop(['H-l', 'H-PC', 'L-PC'], axis=1)
    return df1


def ADX(DF,n):
    "function to calculate ADX"
    df2 = DF.copy()
    df2['TR'] = ATR(df2,n)['TR'] #the period parameter of ATR function does not matter because period does not influence TR calculation
    df2['DMplus']=np.where((df2['High']-df2['High'].shift(1))>(df2['Low'].shift(1)-df2['Low']),df2['High']-df2['High'].shift(1),0)
    df2['DMplus']=np.where(df2['DMplus']<0,0,df2['DMplus'])
    df2['DMminus']=np.where((df2['Low'].shift(1)-df2['Low'])>(df2['High']-df2['High'].shift(1)),df2['Low'].shift(1)-df2['Low'],0)
    df2['DMminus']=np.where(df2['DMminus']<0,0,df2['DMminus'])
    TRn = []
    DMplusN = []
    DMminusN = []
    TR = df2['TR'].tolist()
    DMplus = df2['DMplus'].tolist()
    DMminus = df2['DMminus'].tolist()
    for i in range(len(df2)):
        if i < n:
            TRn.append(np.NaN)
            DMplusN.append(np.NaN)
            DMminusN.append(np.NaN)
        elif i == n:
            TRn.append(df2['TR'].rolling(n).sum().tolist()[n])
            DMplusN.append(df2['DMplus'].rolling(n).sum().tolist()[n])
            DMminusN.append(df2['DMminus'].rolling(n).sum().tolist()[n])
        elif i > n:
            TRn.append(TRn[i-1] - (TRn[i-1]/n) + TR[i])
            DMplusN.append(DMplusN[i-1] - (DMplusN[i-1]/n) + DMplus[i])
            DMminusN.append(DMminusN[i-1] - (DMminusN[i-1]/n) + DMminus[i])
    df2['TRn'] = np.array(TRn)
    df2['DMplusN'] = np.array(DMplusN)
    df2['DMminusN'] = np.array(DMminusN)
    df2['DIplusN']=100*(df2['DMplusN']/df2['TRn'])
    df2['DIminusN']=100*(df2['DMminusN']/df2['TRn'])
    df2['DIdiff']=abs(df2['DIplusN']-df2['DIminusN'])
    df2['DIsum']=df2['DIplusN']+df2['DIminusN']
    df2['DX']=100*(df2['DIdiff']/df2['DIsum'])
    ADX = []
    DX = df2['DX'].tolist()
    for j in range(len(df2)):
        if j < 2*n-1:
            ADX.append(np.NaN)
        elif j == 2*n-1:
            ADX.append(df2['DX'][j-n+1:j+1].mean())
        elif j > 2*n-1:
            ADX.append(((n-1)*ADX[j-1] + DX[j])/n)
    df2['ADX']=np.array(ADX)
    return df2['ADX']

for ticker in tickers_portfolio:
    adx_dict[ticker]['ADX'] = ADX(adx_dict[ticker], 14)
    
    
for ticker in tickers_portfolio:   
    plt.subplot(2, 1, 1)
    plt.plot(adx_dict[ticker].iloc[-100 :, 4])
    plt.title(ticker + ' Average Directional Index')
    
    plt.subplot(2, 1, 2)
    plt.plot(adx_dict[ticker].iloc[-100 :, 6])
    plt.yticks(np.arange(0, 100, 20))
    plt.show()   
    
    
    
    
"""

5. OBV - On balance Volume

"""


obd_dict = {}

for ticker in tickers_portfolio:
    obd_dict[ticker] = work_dict[ticker]
    
    
def OBV(DF):
    df = DF.copy()
    df['Daily_Return'] = df['Adj Close'].pct_change()
    df['Direction'] = np.where(df['Daily_Return'] > 0, 1, -1)
    df['Direction'][0] = 0
    df['Volume_Adjusted'] = df['Volume'] * df['Direction']
    df['OBV'] = df['Volume_Adjusted'].cumsum()
    return df['OBV']

for ticker in tickers_portfolio:
    obd_dict[ticker]['OBV'] = OBV(obd_dict[ticker])
    
for ticker in tickers_portfolio:   
    plt.subplot(2, 1, 1)
    plt.plot(obd_dict[ticker].iloc[-100 :, 4])
    plt.title(ticker + ' On balance Volume')
    
    plt.subplot(2, 1, 2)
    plt.plot(obd_dict[ticker].iloc[-100 :, 7])
    
    plt.show()  



