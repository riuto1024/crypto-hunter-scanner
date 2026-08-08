import json, math
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title='加密猎手·市场扫描器 V1', page_icon='🎯', layout='wide')
st.title('🎯 加密猎手 · 市场扫描器 V1')
st.caption('免费版：市场质量池 → 技术候选池 → TradingView 等 5M / 1M 精确进场')

BINANCE='https://fapi.binance.com'
CG='https://api.coingecko.com/api/v3'

with st.sidebar:
    st.header('入池规则')
    min_cap=st.number_input('最低市值（亿美元）',0.1,1000.0,3.0,0.5)*1e8
    min_spot=st.number_input('最低24H市场成交额（万美元）',100.0,1e8,3000.0,500.0)*1e4
    min_fut=st.number_input('最低24H币安永续成交额（万美元）',100.0,1e8,10000.0,1000.0)*1e4
    max_spread=st.number_input('最大盘口价差（%）',0.01,2.0,0.10,0.01)
    min_amp=st.number_input('最低24H振幅（%）',0.1,50.0,2.0,0.5)
    max_amp=st.number_input('最高24H振幅（%）',2.0,100.0,20.0,1.0)
    pool_n=st.slider('市场质量池显示数量',10,80,30,5)
    tech_n=st.slider('技术扫描前N名',5,25,12,1)
    st.divider()
    st.subheader('免费 CoinGecko')
    st.caption('默认免 Key；如遇限流，可填免费 Demo Key。')
    cg_key=st.text_input('CoinGecko Demo Key（可留空）', type='password')


def get_json(url, params=None, headers=None, timeout=15):
    r=requests.get(url,params=params,headers=headers,timeout=timeout)
    r.raise_for_status(); return r.json()

@st.cache_data(ttl=300, show_spinner=False)
def exchange_info(): return get_json(BINANCE+'/fapi/v1/exchangeInfo')

@st.cache_data(ttl=60, show_spinner=False)
def ticker24(): return get_json(BINANCE+'/fapi/v1/ticker/24hr')

@st.cache_data(ttl=60, show_spinner=False)
def premium(): return get_json(BINANCE+'/fapi/v1/premiumIndex')

@st.cache_data(ttl=300, show_spinner=False)
def cg_markets(key):
    headers={'x-cg-demo-api-key':key} if key else {}
    rows=[]
    for page in (1,2):
        rows += get_json(CG+'/coins/markets', params={'vs_currency':'usd','order':'market_cap_desc','per_page':250,'page':page,'sparkline':'false'}, headers=headers)
    return rows

@st.cache_data(ttl=120, show_spinner=False)
def klines(symbol, interval, limit=230):
    return get_json(BINANCE+'/fapi/v1/klines', params={'symbol':symbol,'interval':interval,'limit':limit})

@st.cache_data(ttl=120, show_spinner=False)
def open_interest(symbol):
    try: return get_json(BINANCE+'/fapi/v1/openInterest', params={'symbol':symbol})
    except Exception: return None


def kdf(raw):
    cols=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_base','taker_quote','ignore']
    d=pd.DataFrame(raw,columns=cols)
    for c in ['open','high','low','close','volume','quote_volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
    return d

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def atr(d,n=14):
    pc=d.close.shift(1)
    tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(span=n,adjust=False).mean()
def clamp(x): return max(0,min(100,x))

def log_score(v,low,high,pts):
    if pd.isna(v) or v<=low:return 0
    if v>=high:return pts
    return pts*math.log(v/low)/math.log(high/low)

def sweet(v,left,ideal_l,ideal_r,right,pts):
    if pd.isna(v) or v<left or v>right:return 0
    if ideal_l<=v<=ideal_r:return pts
    if v<ideal_l:return pts*(v-left)/(ideal_l-left)
    return pts*(right-v)/(right-ideal_r)

def tech_state(symbol):
    d4,d1,d15=[kdf(klines(symbol,i)).iloc[:-1].copy() for i in ['4h','1h','15m']]
    d4['e50']=ema(d4.close,50); d4['e200']=ema(d4.close,200)
    d1['e20']=ema(d1.close,20); d1['e50']=ema(d1.close,50); d1['atr']=atr(d1)
    d15['e20']=ema(d15.close,20); d15['atr']=atr(d15)
    a,b,c=d4.iloc[-1],d1.iloc[-1],d15.iloc[-1]
    direction='多头' if a.close>a.e50>a.e200 else ('空头' if a.close<a.e50<a.e200 else '等待')
    context='等待'
    if direction=='多头':
        if b.close<b.e50: context='风险'
        elif b.e20>b.e50: context='回调' if abs(b.close-b.e20)<=b.atr else '趋势'
    elif direction=='空头':
        if b.close>b.e50: context='风险'
        elif b.e20<b.e50: context='回调' if abs(b.close-b.e20)<=b.atr else '趋势'
    zl,zh=c.e20-.5*c.atr,c.e20+.5*c.atr
    if zl<=c.close<=zh and direction=='多头': zone='多头区域'
    elif zl<=c.close<=zh and direction=='空头': zone='空头区域'
    else:
        dist=abs(c.close-c.e20)/max(c.atr,1e-9)
        zone='接近区域' if dist<=1 and direction!='等待' else '等待'
    score=(35 if direction!='等待' else 0)+(35 if context=='回调' else 20 if context=='趋势' else 0)+(30 if '区域' in zone and zone!='接近区域' else 15 if zone=='接近区域' else 0)
    if context=='风险': score=0
    return {'4H方向':direction,'1H状态':context,'15M位置':zone,'技术机会分':int(clamp(score))}

try:
    with st.spinner('正在读取免费市场数据……'):
        ex=exchange_info(); t=ticker24(); p=premium(); cg=cg_markets(cg_key.strip())

    perps={s['symbol']:s.get('baseAsset','') for s in ex.get('symbols',[]) if s.get('status')=='TRADING' and s.get('quoteAsset')=='USDT' and s.get('contractType')=='PERPETUAL'}
    td=pd.DataFrame(t)
    for c in ['lastPrice','quoteVolume','highPrice','lowPrice','priceChangePercent','bidPrice','askPrice']:
        if c in td: td[c]=pd.to_numeric(td[c],errors='coerce')
    td=td[td.symbol.isin(perps)].copy(); td['base']=td.symbol.map(perps)

    pdm=pd.DataFrame(p)
    if not pdm.empty and 'lastFundingRate' in pdm:
        pdm['lastFundingRate']=pd.to_numeric(pdm.lastFundingRate,errors='coerce')
        td=td.merge(pdm[['symbol','lastFundingRate']].drop_duplicates('symbol'),on='symbol',how='left')
    else: td['lastFundingRate']=np.nan

    cd=pd.DataFrame(cg)
    cd['symbol_key']=cd.symbol.astype(str).str.upper()
    cd=cd.sort_values('market_cap',ascending=False).drop_duplicates('symbol_key')
    cd=cd[['symbol_key','name','market_cap','total_volume','market_cap_rank']]
    df=td.merge(cd,left_on='base',right_on='symbol_key',how='inner')
    df=df[~df.base.isin({'USDC','USDE','FDUSD','TUSD','DAI','USDP','BUSD','EUR','TRY','BRL'})].copy()
    df['换手率']=df.total_volume/df.market_cap.replace(0,np.nan)
    df['永续成交额']=df.quoteVolume
    df['24H振幅%']=(df.highPrice-df.lowPrice)/df.lastPrice.replace(0,np.nan)*100
    if 'bidPrice' in df and 'askPrice' in df:
        mid=(df.bidPrice+df.askPrice)/2
        df['盘口价差%']=(df.askPrice-df.bidPrice)/mid.replace(0,np.nan)*100
    else: df['盘口价差%']=np.nan
    df['资金费率%']=df.lastFundingRate.fillna(0)*100

    hard=(df.market_cap>=min_cap)&(df.total_volume>=min_spot)&(df['永续成交额']>=min_fut)&(df['24H振幅%']>=min_amp)&(df['24H振幅%']<=max_amp)
    if df['盘口价差%'].notna().any(): hard &= df['盘口价差%'].fillna(999)<=max_spread
    pool=df[hard].copy()

    def quality(r):
        s=0
        s+=log_score(r.market_cap,3e8,5e10,10)
        s+=log_score(r['永续成交额'],1e8,5e9,20)
        s+=log_score(r.total_volume,3e7,3e9,15)
        s+=sweet(r['换手率'],.005,.03,.35,1.2,15)
        s+=sweet(r['24H振幅%'],1.5,3,10,22,15)
        sp=r['盘口价差%']
        if pd.notna(sp): s+=15 if sp<=.02 else 12 if sp<=.05 else 7 if sp<=.10 else 0
        f=abs(r['资金费率%']); s+=10 if f<=.02 else 7 if f<=.05 else 3 if f<=.10 else 0
        return int(round(clamp(s)))

    if not pool.empty:
        pool['市场质量分']=pool.apply(quality,axis=1)
        pool=pool.sort_values(['市场质量分','永续成交额','market_cap'],ascending=False).reset_index(drop=True)

    st.subheader('一、市场质量池')
    a,b,c,d=st.columns(4)
    a.metric('币安USDT永续',len(perps)); b.metric('完成市值匹配',len(df)); c.metric('硬条件合格',len(pool)); d.metric('重点显示',min(pool_n,len(pool)))
    if pool.empty:
        st.warning('当前没有币种满足条件，可在左侧降低门槛。'); st.stop()

    show=pool.head(pool_n).copy(); show['排名']=range(1,len(show)+1)
    show['市值(亿$)']=show.market_cap/1e8; show['24H市场成交(亿$)']=show.total_volume/1e8; show['永续成交(亿$)']=show['永续成交额']/1e8; show['换手率%']=show['换手率']*100
    out=show[['排名','base','name','市场质量分','市值(亿$)','24H市场成交(亿$)','永续成交(亿$)','换手率%','24H振幅%','盘口价差%','资金费率%']].rename(columns={'base':'币种','name':'名称'})
    st.dataframe(out,use_container_width=True,hide_index=True,column_config={
        '市场质量分':st.column_config.ProgressColumn(min_value=0,max_value=100,format='%d'),
        '市值(亿$)':st.column_config.NumberColumn(format='%.2f'),'24H市场成交(亿$)':st.column_config.NumberColumn(format='%.2f'),'永续成交(亿$)':st.column_config.NumberColumn(format='%.2f'),
        '换手率%':st.column_config.NumberColumn(format='%.2f%%'),'24H振幅%':st.column_config.NumberColumn(format='%.2f%%'),'盘口价差%':st.column_config.NumberColumn(format='%.4f%%'),'资金费率%':st.column_config.NumberColumn(format='%.4f%%')})
    st.info('市场质量分只代表“适不适合日内交易”，不是胜率，也不是买卖方向。')

    st.subheader('二、Hunter 技术候选池')
    st.caption('只扫描市场质量排名靠前的币，减少免费API请求。')
    if st.button(f'🎯 扫描前 {min(tech_n,len(pool))} 名技术状态',type='primary',use_container_width=True):
        rows=[]; cand=pool.head(tech_n); prog=st.progress(0)
        for i,(_,r) in enumerate(cand.iterrows(),1):
            try:
                tech=tech_state(r.symbol); oi=open_interest(r.symbol); oiv=np.nan
                if oi and oi.get('openInterest'): oiv=float(oi['openInterest'])*float(r.lastPrice)/1e6
                rows.append({'币种':r.base,'市场质量分':int(r['市场质量分']),**tech,'OI估算(百万$)':oiv,'资金费率%':r['资金费率%']})
            except Exception:
                rows.append({'币种':r.base,'市场质量分':int(r['市场质量分']),'4H方向':'数据失败','1H状态':'-','15M位置':'-','技术机会分':0,'OI估算(百万$)':np.nan,'资金费率%':r['资金费率%']})
            prog.progress(i/len(cand))
        res=pd.DataFrame(rows); res['综合关注分']=(res['市场质量分']*.55+res['技术机会分']*.45).round().astype(int)
        res['状态']=res.apply(lambda x:'★★★ 重点' if x['技术机会分']>=80 and x['市场质量分']>=75 else ('★★ 观察' if x['技术机会分']>=65 else ('★ 接近' if x['技术机会分']>=50 else '等待')),axis=1)
        res=res.sort_values(['综合关注分','技术机会分','市场质量分'],ascending=False).reset_index(drop=True)
        st.dataframe(res,use_container_width=True,hide_index=True,column_config={'市场质量分':st.column_config.ProgressColumn(min_value=0,max_value=100,format='%d'),'技术机会分':st.column_config.ProgressColumn(min_value=0,max_value=100,format='%d'),'综合关注分':st.column_config.ProgressColumn(min_value=0,max_value=100,format='%d'),'OI估算(百万$)':st.column_config.NumberColumn(format='%.1f'),'资金费率%':st.column_config.NumberColumn(format='%.4f%%')})
        best=res[(res['技术机会分']>=65)&res['4H方向'].isin(['多头','空头'])]
        if len(best): st.success('建议打开 TradingView 继续等待 5M / 1M 的候选：'+'、'.join(best.币种.head(8).tolist()))
        else: st.warning('当前前排币种还没有达到技术候选标准，不需要勉强交易。')

except requests.HTTPError as e:
    st.error(f'行情接口请求失败：{e}')
    st.caption('免费接口可能短暂限流；稍后刷新，或填写免费的 CoinGecko Demo Key。')
except Exception as e:
    st.error(f'程序运行失败：{e}')

st.divider(); st.caption('V1原则：先免费验证。市场质量评分与技术机会评分分开，最终进场仍交给 TradingView 的“加密猎手·精准交易”。')
