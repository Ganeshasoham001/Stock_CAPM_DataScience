import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import pandas_datareader.data as pdr
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
st.set_page_config(page_title= "CAPM",
                   page_icon= "chart_with_upward_trends",
                   layout= 'wide')

st.title("Cpital Asset Pricing Model")
#user input
col1,col2=st.columns([1,1])
with col1:
    stocks_list=st.multiselect("Choose 4 stocks",('TSLA','AAPL','NFLX','MSFT','AMZN','NVDA','GOOGL','GS','INTC'),['TSLA','GOOGL','INTC','GS'])

with col2:
    year=st.number_input("Number of years",1,10)
#download data from SP500
try:
    end= dt.date.today()
    start= dt.date(dt.date.today().year-year,dt.date.today().month,dt.date.today().day)
    SP500= pdr.DataReader(['sp500'],'fred',start,end)
    stocks_df=pd.DataFrame()
    for stock in stocks_list:
        data=yf.download(stock,period= f'{year}y')
        stocks_df[f'{stock}']= data['Close']
    stocks_df.reset_index(inplace=True)
    SP500.reset_index(inplace=True)
    SP500.columns= ['Date','sp500']
    stocks_df['Date'] = stocks_df['Date'].astype('datetime64[ns]')
    stocks_df['Date']= stocks_df['Date'].apply(lambda x : str(x)[:10])
    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
    stocks_df= pd.merge(stocks_df,SP500,on='Date',how='inner')
    print(stocks_df)
    col1,col2= st.columns([1,1])
    with col1:
        st.markdown("### Dtaframe head")
        st.dataframe(stocks_df.head(),use_container_width=True)
    with col2:
        st.markdown("### Dtaframe tail")
        st.dataframe(stocks_df.tail(),use_container_width=True)

    col1,col2 = st.columns([1,1])
    with col1:
        st.markdown("### Price of all stocks")
        st.plotly_chart(interactive_plot(stocks_df))
    with col2:                                                              
        st.markdown('### Price of All Stocks after Normalizing')
        st.plotly_chart(interactive_plot(normalized(stocks_df)))
    stocks_daily_return=daily_returns(stocks_df)
    print(stocks_daily_return.head())



    beta={}
    alpha={}
    for i in stocks_daily_return.columns:
        if i !='Date' and i!='sp500':
            b,a =calculate_beta(stocks_daily_return,i)

            beta[i]=b
            alpha[i]=a
    print(beta,alpha)
    beta_df=pd.DataFrame()
    beta_df['stock']=beta.keys()
    beta_df['Beta Values'] = [str (round(i,2)) for i in beta.values()]

    col1,col2 = st.columns([1,1])
    with col1:
        st.markdown("### Calculate Beta ")
        st.dataframe(beta_df, use_container_width=True)

    rf=0
    rm=stocks_daily_return['sp500'].mean()*252
    return_df= pd.DataFrame()
    return_values=[]
    for stock,value in beta.items():
        return_values.append(str(round(rf+(value*(rm-rf)),2)))
    return_df['Stock']= stocks_list
    return_df['Return Value'] = return_values
    with col2:
        st.markdown("### Calculate return using CAPM")
        st.dataframe(return_df,use_container_width=True)
except:
    st.write('Please select Valid stocks')
