"""
utils.py — helper functions for the RAG Data-Analyst API
• scrape_table_from_url()  →  [DataFrame, list-of-lists]
• run_duckdb_query()       →  list[dict]
• plot_and_encode_base64() →  PNG data-URI
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import re
import base64
import io
import requests
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------
# 0. Column-name sanitiser for DuckDB
# ---------------------------------------------------------------------
_SQL_RESERVED = {
    "rank", "order", "group", "limit", "offset", "where",
    "select", "table", "year", "month"
}

def _sql_safe(col: str) -> str:
    """Make a column name safe for SQL (DuckDB)."""
    col_clean = re.sub(r"\W+", "_", col.strip())          # spaces / punct → _
    if col_clean.lower() in _SQL_RESERVED:               # avoid reserved words
        col_clean += "_"
    return re.sub(r"__+", "_", col_clean).strip("_")

# ---------------------------------------------------------------------
# 1. HTML table scraper
# ---------------------------------------------------------------------
def scrape_table_from_url(url: str):
    """
    Return a list with **two** parallel views of the first HTML table:

        tables[0] -> pandas.DataFrame
                     (contains BOTH original and SQL-safe alias columns)
        tables[1] -> list-of-lists  [header_row, row1, row2, …]

    Pandas code can keep using the original names (“Worldwide gross”),
    while DuckDB SQL can use the aliases (“Worldwide_gross”, “Rank_”, …).
    """
    # 1️⃣  Fast path: pandas read_html (lxml)
    try:
        df = pd.read_html(url, flavor="lxml")[0]
    except Exception:
        # 2️⃣  Fallback: BeautifulSoup manual parse
        resp = requests.get(url, timeout=30)
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")

        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        rows = []
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            # pad / trim so every row length == header length
            if len(cells) < len(headers):
                cells += [""] * (len(headers) - len(cells))
            elif len(cells) > len(headers):
                cells = cells[: len(headers)]
            rows.append(cells)

        df = pd.DataFrame(rows, columns=headers)

    # 3️⃣  Add SQL-safe aliases
    original_cols = df.columns.tolist()
    safe_cols = [_sql_safe(c) for c in original_cols]

    for orig, safe in zip(original_cols, safe_cols):
        if orig != safe and safe not in df.columns:
            df[safe] = df[orig]          # duplicate column as alias

    # 4️⃣  Build list-of-lists using original header order
    list_version = [original_cols] + df[original_cols].values.tolist()

    return [df, list_version]

# ---------------------------------------------------------------------
# 2. DuckDB query helper
# ---------------------------------------------------------------------
def run_duckdb_query(query: str, files: dict | None = None):
    """
    Execute a DuckDB SQL query.

    Parameters
    ----------
    query : str
        SQL string.
    files : dict | None
        Mapping name → DataFrame **or** file path (e.g. parquet).
        DataFrames are registered directly; paths are exposed via a VIEW.

    Returns
    -------
    list[dict]
        Query results as list of dictionaries.
    """
    con = duckdb.connect()
    try:
        if files:
            for name, obj in files.items():
                if isinstance(obj, pd.DataFrame):
                    con.register(name, obj)
                else:  # assume string path
                    con.execute(
                        f"CREATE VIEW {name} AS "
                        f"SELECT * FROM read_parquet('{obj}')"
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
    Accept a matplotlib Figure *or* the plt module; return data-URI string.
    """
    if not hasattr(fig, "savefig"):      # caller passed `plt`
        fig = fig.gcf()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"

# utils.py  – ADD THIS BLOCK
import pandas as pd   # already imported at top, just shown for context

# ---------------------------------------------------------------------
#  Numeric-cleaning helper (new)
# ---------------------------------------------------------------------
def to_float(series):
    """
    Convert a pandas Series of currency / numeric strings to float.
    • Strips '$' and commas
    • Uses errors='coerce' so bad values become NaN instead of crashing
    """
    return pd.to_numeric(series.replace(r'[\$,]', '', regex=True),
                         errors='coerce')

