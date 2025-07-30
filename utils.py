"""
utils.py – helper functions for the RAG Data Analyst Agent
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import base64
import io


# --------------------------------------------------------------------------------------
# 1) Web-table scraper (now safer)
# --------------------------------------------------------------------------------------
def scrape_table_from_url(url: str, timeout: int = 15):
    """
    Download *all* HTML tables from a URL and return them as a list of pandas
    DataFrames.  Raises ValueError if no table can be parsed.

    • Uses BeautifulSoup so we’re not at the mercy of pandas guessing tags.
    • Silently skips any <table> that pandas.read_html() can’t parse.
    """

    # -- 1. Fetch page -----------------------------------------------------------------
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; RAG-Data-Analyst/1.0)"
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()                               # HTTP-level failure ⇒ exception

    # -- 2. Extract tables -------------------------------------------------------------
    soup = BeautifulSoup(resp.text, "html.parser")
    raw_tables = soup.find_all("table")
    parsed_tables = []

    for tbl in raw_tables:
        try:
            df = pd.read_html(str(tbl), flavor="bs4")[0]
            parsed_tables.append(df)
        except Exception:
            # Skip malformed / non-data tables
            continue

    if not parsed_tables:
        raise ValueError(f"No HTML tables could be parsed at {url}")

    return parsed_tables


# --------------------------------------------------------------------------------------
# 2) DuckDB helper
# --------------------------------------------------------------------------------------
def run_duckdb_query(query: str, files: dict | None = None):
    """
    Run a DuckDB SQL query and return a list of row-dicts.
    Optionally register Parquet/CSV files first:

        run_duckdb_query("SELECT * FROM my_tbl", files={"my_tbl": "data/my.parquet"})
    """
    con = duckdb.connect()
    if files:
        for name, path in files.items():
            con.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')"
            )
    res = con.execute(query).fetchall()
    cols = [d[0] for d in con.description]
    return [dict(zip(cols, row)) for row in res]


# --------------------------------------------------------------------------------------
# 3) Matplotlib → base64 helper
# --------------------------------------------------------------------------------------
def plot_and_encode_base64(fig) -> str:
    """
    Encode a Matplotlib figure as a `data:image/png;base64,...` string.
    Closes the figure after encoding to free memory.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"
