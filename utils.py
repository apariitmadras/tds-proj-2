"""
Utility helpers for the RAG Data-Analyst API
- scrape_table_from_url()  →  [DataFrame, list-of-lists]
- run_duckdb_query()       →  list[dict]
- plot_and_encode_base64() →  PNG data-URI
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import requests
import pandas as pd
from bs4 import BeautifulSoup

import duckdb
import matplotlib.pyplot as plt
import base64, io

# ---------------------------------------------------------------------
# 1. HTML table scraper
# ---------------------------------------------------------------------
def scrape_table_from_url(url: str):
    """
    Fetch the first <table> on the page and return TWO parallel formats:

        tables[0]  ->  pandas.DataFrame        (ideal for analysis)
        tables[1]  ->  list-of-lists
                       [ header_row, row1, row2, ... ]
                       (works for code that rebuilds its own DataFrame)

    If pandas.read_html fails (e.g. html5lib not installed) we fall back
    to BeautifulSoup parsing and guarantee row-length consistency.
    """
    try:
        # pandas uses lxml by default; html5lib is optional but nicer.
        df = pd.read_html(url, flavor="lxml")[0]
    except Exception:
        # ------------------------- Fallback ---------------------------
        resp = requests.get(url, timeout=30)
        soup = BeautifulSoup(resp.text, "html.parser")

        table = soup.find("table")
        headers = [th.get_text(strip=True) for th in table.find_all("th")]

        rows = []
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]

            # 🔧 pad / trim so every row matches header length
            if len(cells) < len(headers):
                cells += [""] * (len(headers) - len(cells))
            elif len(cells) > len(headers):
                cells = cells[: len(headers)]

            rows.append(cells)

        df = pd.DataFrame(rows, columns=headers)

    # Build the list-of-lists version
    list_version = [df.columns.tolist()] + df.values.tolist()

    # Return both so caller can choose
    return [df, list_version]

# ---------------------------------------------------------------------
# 2. DuckDB helper
# ---------------------------------------------------------------------
def run_duckdb_query(query: str, files: dict | None = None):
    """
    Execute a DuckDB SQL query.
      files = {"view_name": "/path/to/parquet"}  (optional)
    """
    con = duckdb.connect()
    try:
        if files:
            for name, path in files.items():
                con.execute(
                    f"CREATE VIEW {name} AS "
                    f"SELECT * FROM read_parquet('{path}')"
                )
        result = con.execute(query).fetchall()
        columns = [desc[0] for desc in con.description]
        return [dict(zip(columns, row)) for row in result]
    finally:
        con.close()

# ---------------------------------------------------------------------
# 3. Figure → base64 PNG
# ---------------------------------------------------------------------
def plot_and_encode_base64(fig):
    """
    Accept either a matplotlib Figure OR the `plt` module,
    and return a base-64 PNG data URI.
    """
    if not hasattr(fig, "savefig"):      # user passed `plt`
        fig = fig.gcf()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"
