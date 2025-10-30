# europe_factor_universe_pipeline.py
# Downloads prices for a 135-stock European universe, cleans data, and writes prices/returns CSVs.
# Usage: python europe_factor_universe_pipeline.py

import logging, os, sys, time
from datetime import datetime
import pandas as pd
import yfinance as yf

TICKER_MAP = [('HSBA', 'HSBA.L'), ('ALV', 'ALV.DE'), ('SAN', 'SAN.MC'), ('UBSG', 'UBSG.SW'), ('BBVA', 'BBVA.MC'), ('UCG', 'UCG.MI'), ('ISP', 'ISP.MI'), ('CS', 'CS.PA'), ('BNP', 'BNP.PA'), ('INGA', 'INGA.AS'), ('BARC', 'BARC.L'), ('CABK', 'CABK.MC'), ('LLOY', 'LLOY.L'), ('DBK', 'DBK.DE'), ('LSEG', 'LSEG.L'), ('INVE.B', 'INVE-B.ST'), ('NDA.FI', 'NDA-FI.HE'), ('NDASE', 'NDA-SE.ST'), ('G', 'G.MI'), ('NWG', 'NWG.L'), ('ACA', 'ACA.PA'), ('III', 'III.L'), ('SREN', 'SREN.SW'), ('GLE', 'GLE.PA'), ('STAN', 'STAN.L'), ('EQT', 'EQT.ST'), ('SEB.A', 'SEB-A.ST'), ('CBK', 'CBK.DE'), ('PRU', 'PRU.L'), ('SWED.A', 'SWED-A.ST'), ('AV.', 'AV.L'), ('VNA', 'VNA.DE'), ('SHB.A', 'SHB-A.ST'), ('BMPS', 'BMPS.MI'), ('ABN', 'ABN.AS'), ('BPE', 'BPE.MI'), ('BAMI', 'BAMI.MI'), ('NN', 'NN.AS'), ('LGEN', 'LGEN.L'), ('CVC', 'CVC.AS'), ('SAB1', 'SAB.MC'), ('UNI', 'UNI.MC'), ('BMED', 'BMED.MI'), ('MAP', 'MAP.MC'), ('FBK', 'FBK.MI'), ('BKT', 'BKT.MC'), ('SGRO', 'SGRO.L'), ('AGN', 'AGN.AS'), ('LI', 'LI.PA'), ('STJ', 'STJ.L'), ('MRL', 'MRL.MC'), ('PHNX', 'PHNX.L'), ('BALD.B', 'BALD-B.ST'), ('MNG', 'MNG.L'), ('SDR', 'SDR.L'), ('ICG', 'ICG.L'), ('BEZ', 'BEZ.L'), ('UNI1', 'UNI.MC'), ('LAND', 'LAND.L'), ('LMP', 'LMP.L'), ('HSX', 'HSX.L'), ('CAST', 'CAST.ST'), ('AT1', 'AT1.DE'), ('BBOX', 'BBOX.L'), ('AZM', 'AZM.MI'), ('BLND', 'BLND.L'), ('IGG', 'IGG.L'), ('INVP', 'INVP.L'), ('ABDN', 'ABDN.L'), ('NOBA', 'NOBA.ST'), ('PHLL', 'PHLL.L'), ('ALLFG', 'ALLFG.AS'), ('COL', 'COL.MC'), ('UTG', 'UTG.L'), ('318', 'BOCH.DE'), ('SHC', 'SHC.L'), ('QLT', 'QLT.L'), ('BPT', 'BPT.L'), ('PHP', 'PHP.L'), ('DN3', 'DN3.F'), ('IWG', 'IWG.L'), ('EMG', 'EMG.L'), ('FABG', 'FABG.ST'), ('JUST', 'JUST.L'), ('AJB', 'AJB.L'), ('WALL.B', 'WALL-B.ST'), ('OSB', 'OSB.L'), ('TCAP', 'TCAP.L'), ('KINV.B', 'KINV-B.ST'), ('HMSO', 'HMSO.L'), ('SRE', 'SRE.L'), ('GRI', 'GRI.L'), ('N91', 'N91.L'), ('GPE', 'GPE.L'), ('ASHM', 'ASHM.L'), ('YCA', 'YCA.L'), ('STOR.B', 'STOR-B.ST'), ('SUPR', 'SUPR.L'), ('GROW', 'GROW.L'), ('JUP', 'JUP.L'), ('MTRO', 'MTRO.L'), ('CBG', 'CBG.L'), ('SBB.B', 'SBB-B.ST'), ('CSN', 'CSN.L'), ('PRSR', 'PRSR.L'), ('LOGI.B', 'LOGI-B.ST'), ('CHRY', 'CHRY.L'), ('THRL', 'THRL.L'), ('CRE', 'CRE.L'), ('ESP', 'ESP.L'), ('INTRUM', 'INTRUM.ST'), ('IPO', 'IPO.L'), ('DOV1', 'DOV1.MI'), ('IPF', 'IPF.L'), ('CORE.B', 'CORE-B.ST'), ('PCTN', 'PCTN.L'), ('FCH', 'FCH.L'), ('CREI', 'CREI.L'), ('NRR', 'NRR.L'), ('VANQ', 'VANQ.L'), ('SOHO', 'SOHO.L'), ('SREI', 'SREI.L'), ('CLI', 'CLI.L'), ('SSIT', 'SSIT.L'), ('VEFAB', 'VEFAB.ST'), ('FOXT', 'FOXT.L'), ('ASLI', 'ASLI.L'), ('WJG', 'WJG.L'), ('ALCBI', 'ALCBI.PA'), ('DOXA', 'DOXA.ST'), ('ARCA', 'ARCA.ST'), ('ZEST', 'ZEST.MC'), ('API', 'API.L'), ('LIT', 'LIT.L'), ('BTCX', 'BTCX.ST')]
YAHOO_TICKERS = [y for _, y in TICKER_MAP]

START_DATE = "2013-01-01"   # adjust if needed
END_DATE = None             # None = up to latest available
CHUNK_SIZE = 60             # download in chunks to be safe
MIN_HISTORY_DAYS = 500      # drop symbols with too little history

OUT_DIR = "Investment_universe"
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(OUT_DIR, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def download_chunk(tickers):
    for attempt in range(3):
        try:
            logging.info(f"Downloading {len(tickers)} tickers... attempt {attempt+1}")
            df = yf.download(
                tickers,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,
                threads=True,
                group_by="ticker"
            )

            # handle if df is a multi-index from multiple tickers
            if isinstance(df.columns, pd.MultiIndex):
                df = df.loc[:, pd.IndexSlice[:, "Close"]]
                df.columns = [c[0] for c in df.columns]

            # handle single ticker returns
            elif "Close" in df.columns:
                df = df[["Close"]].rename(columns={"Close": tickers[0]})

            return df
        except Exception as e:
            logging.warning(f"Download failed (attempt {attempt+1}): {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("Failed to download after 3 attempts.")

def main():
    logging.info(f"Universe size: {len(YAHOO_TICKERS)}")
    frames = []
    for tick_chunk in chunked(YAHOO_TICKERS, CHUNK_SIZE):
        df = download_chunk(tick_chunk)
        frames.append(df)

    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.dropna(axis=1, how="all")

    # Filter by minimum history
    long_enough = [c for c in prices.columns if prices[c].dropna().shape[0] >= MIN_HISTORY_DAYS]
    dropped = sorted(set(prices.columns) - set(long_enough))
    if dropped:
        logging.info(f"Dropping {len(dropped)} tickers with <{MIN_HISTORY_DAYS} days: {dropped}")
    prices = prices[long_enough]

    # Fill small gaps
    prices = prices.ffill().bfill()

    # Save prices
    prices_path = os.path.join(OUT_DIR, "europe_prices.csv")
    prices.to_csv(prices_path)
    logging.info(f"Wrote prices to {prices_path} with shape {prices.shape}")

    # Compute daily returns
    returns = prices.pct_change().dropna(how="all")
    returns_path = os.path.join(OUT_DIR, "europe_returns.csv")
    returns.to_csv(returns_path)
    logging.info(f"Wrote returns to {returns_path} with shape {returns.shape}")

    # Save the mapping for the final columns
    present = [c for c in prices.columns]
    present_map = [(r,y) for (r,y) in TICKER_MAP if y in present]
    pd.DataFrame(present_map, columns=["raw_ticker","yahoo_symbol"]).to_csv(
        os.path.join(OUT_DIR, "universe_final_mapping.csv"), index=False
    )
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
