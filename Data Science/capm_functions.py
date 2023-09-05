import plotly.express as px
import numpy as np
def interactive_plot(df):
    fig= px.line()
    for i in df.columns[1:]:
        fig.add_scatter(x=df['Date'], y= df[i],name= i)
    fig.update_layout(width=450,margin=dict(l=20,r=20,t=60,b=20),legend= dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x =1,))
    return fig
# function for normalize of prices
def normalized(df_2):
    df=df_2.copy()
    for i in df.columns[1:]:
        df[i]=df[i]/df[i][0]
    return df
#functions to calculate daily returns
def daily_returns(df):
    df_daily_rtr=df.copy()
    for i in df.columns[1:]:
        for j in range(1,len(df)):
            df_daily_rtr[i][j]= ((df[i][j]-df[i][j-1])/df[i][j-1])*100
        df_daily_rtr[i][0]=0
    return df_daily_rtr
 


def calculate_beta(stocks_daily_rtr,stock):
    rm= stocks_daily_rtr['sp500'].mean()*252
    b,a=np.polyfit(stocks_daily_rtr['sp500'],stocks_daily_rtr[stock],1)
    return b,a
