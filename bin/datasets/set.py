import pandas as pd
import pandas_datareader.data as web

def clean_ff(df, is_daily=False):
    df = df.copy()
    df.columns = [c.strip().replace(' ', '').replace('-', '_') for c in df.columns]
    # convert percent to decimals
    df = df / 100.0
    if not is_daily:
        # monthly PeriodIndex -> month-end timestamp
        if hasattr(df.index, "to_timestamp"):
            df.index = df.index.to_timestamp(how="end")
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df

def fetch_ff5_plus_mom():
    ff5_m = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start=1963)[0]
    ff5_d = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench', start=1963)[0]
    mom_m = web.DataReader('F-F_Momentum_Factor', 'famafrench', start=1926)[0]  # MOM goes further back
    mom_d = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench', start=1963)[0]

    ff5_m = clean_ff(ff5_m, is_daily=False).rename(columns={'Mkt_RF':'MKT_RF'})
    ff5_d = clean_ff(ff5_d, is_daily=True).rename(columns={'Mkt_RF':'MKT_RF'})
    mom_m = clean_ff(mom_m, is_daily=False).rename(columns={'Mom':'MOM'})
    mom_d = clean_ff(mom_d, is_daily=True).rename(columns={'Mom':'MOM'})

    ff5_m = ff5_m.join(mom_m[['MOM']], how='left')
    ff5_d = ff5_d.join(mom_d[['MOM']], how='left')
    return ff5_m, ff5_d

def fetch_ff3_plus_mom():
    ff3_m = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=1926)[0]       # MKT-RF, SMB, HML, RF
    ff3_d = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start=1963)[0] # daily only from 1963
    mom_m = web.DataReader('F-F_Momentum_Factor', 'famafrench', start=1926)[0]
    mom_d = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench', start=1963)[0]

    ff3_m = clean_ff(ff3_m, is_daily=False).rename(columns={'Mkt_RF':'MKT_RF'})
    ff3_d = clean_ff(ff3_d, is_daily=True).rename(columns={'Mkt_RF':'MKT_RF'})
    mom_m = clean_ff(mom_m, is_daily=False).rename(columns={'Mom':'MOM'})
    mom_d = clean_ff(mom_d, is_daily=True).rename(columns={'Mom':'MOM'})

    ff3_m = ff3_m.join(mom_m[['MOM']], how='left')  # gives you 1926+ with MOM
    ff3_d = ff3_d.join(mom_d[['MOM']], how='left')
    return ff3_m, ff3_d

# ---- Fetch all ----
ff5_m, ff5_d = fetch_ff5_plus_mom()
ff3_m, ff3_d = fetch_ff3_plus_mom()

# ---- Save ----
ff5_m.to_csv('ff5_plus_mom_monthly.csv', index_label='Date')
ff5_d.to_csv('ff5_plus_mom_daily.csv', index_label='Date')
ff3_m.to_csv('ff3_plus_mom_monthly.csv', index_label='Date')
ff3_d.to_csv('ff3_plus_mom_daily.csv', index_label='Date')

print("✅ Saved:")
print(f" - ff5_plus_mom_monthly.csv (rows {len(ff5_m)}, start {ff5_m.index.min().date()}, end {ff5_m.index.max().date()})")
print(f" - ff5_plus_mom_daily.csv   (rows {len(ff5_d)}, start {ff5_d.index.min().date()}, end {ff5_d.index.max().date()})")
print(f" - ff3_plus_mom_monthly.csv (rows {len(ff3_m)}, start {ff3_m.index.min().date()}, end {ff3_m.index.max().date()})")
print(f" - ff3_plus_mom_daily.csv   (rows {len(ff3_d)}, start {ff3_d.index.min().date()}, end {ff3_d.index.max().date()})")
