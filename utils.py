"""
utils.py – helper functions for the RAG Data-Analyst API
* scrape_table_from_url()  →  [DataFrame, list-of-lists]
* to_float(series)         →  robust currency / numeric converter
* run_duckdb_query()       →  list[dict]
* plot_and_encode_base64() →  PNG data-URI
"""
# ------------------------------------------------------------------
import re, base64, io, requests, pandas as pd, duckdb, matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# ------------------------------------------------------------------
# 0.  numeric-clean helper  (LLM will call this)
# ------------------------------------------------------------------
def to_float(series):
    """
    Strip $, commas, *and any letter-footnote markers* and return float.
    Bad cells → NaN (errors='coerce').
    """
    cleaned = series.replace(r'[A-Za-z]+\d*', '', regex=True)   # kill “T”, “SM”, etc.
    cleaned = cleaned.replace(r'[\$,]', '', regex=True)
    return pd.to_numeric(cleaned, errors='coerce')

# ------------------------------------------------------------------
# 1.  scrape first HTML table → DataFrame + list-of-lists
# ------------------------------------------------------------------
def scrape_table_from_url(url: str):
    """Return [tbl_df, list_version] for the first <table> on the page."""
    try:
        # fast path – pandas read_html with lxml
        tbl_df = pd.read_html(url, flavor="lxml")[0]
    except Exception:
        # fallback – BeautifulSoup manual parse
        resp  = requests.get(url, timeout=30)
        soup  = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        rows = []
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all(["td","th"])]
            cells = (cells + [""] * len(headers))[: len(headers)]   # pad/trim
            rows.append(cells)
        tbl_df = pd.DataFrame(rows, columns=headers)

    # list-of-lists form (original header order)
    list_version = [tbl_df.columns.tolist()] + tbl_df.values.tolist()
    return [tbl_df, list_version]

# ------------------------------------------------------------------
# 2.  DuckDB helper (unchanged API, but now accepts DataFrames)
# ------------------------------------------------------------------
def run_duckdb_query(query: str, files: dict | None = None):
    con = duckdb.connect()
    try:
        if files:
            for name, obj in files.items():
                if isinstance(obj, pd.DataFrame):
                    con.register(name, obj)
                else:  # assume path
                    con.execute(
                        f"CREATE VIEW {name} AS "
                        f"SELECT * FROM read_parquet('{obj}')"
                    )
        res     = con.execute(query).fetchall()
        columns = [d[0] for d in con.description]
        return [dict(zip(columns, row)) for row in res]
    finally:
        con.close()

# ------------------------------------------------------------------
# 3.  Matplotlib figure → base64 PNG
# ------------------------------------------------------------------
def plot_and_encode_base64(fig):
    if not hasattr(fig, "savefig"):  # user passed plt
        fig = fig.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"
