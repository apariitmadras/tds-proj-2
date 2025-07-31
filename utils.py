"""
Utility helpers for the RAG Data-Analyst API
- scrape_table_from_url()  →  [DataFrame, list-of-lists] with safe column names
- run_duckdb_query()       →  list[dict]
- plot_and_encode_base64() →  PNG data-URI
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import re, base64, io, requests, duckdb
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------
# 1. HTML table scraper  +  column-name sanitiser
# ---------------------------------------------------------------------
_SQL_RESERVED = {
    # a handful that appear often; add more as needed
    "rank", "order", "group", "limit", "offset", "where",
    "select", "table", "year", "month"
}

def _sql_safe(col: str) -> str:
    """
    Make a column name safe for DuckDB:
    - spaces / punctuation  →  underscore
    - if it’s a reserved keyword, append an underscore
    - collapse multiple underscores
    """
    col_clean = re.sub(r"\W+", "_", col.strip())
    if col_clean.lower() in _SQL_RESERVED:
        col_clean += "_"
    return re.sub(r"__+", "_", col_clean).strip("_")

def scrape_table_from_url(url: str):
    """
    Return a list with two parallel representations of the first HTML table:
      tables[0] = pandas DataFrame  (with SQL-safe column names)
      tables[1] = list-of-lists     [ header_row, row1, row2, … ]
    """
    try:
        df = pd.read_html(url, flavor="lxml")[0]
    except Exception:
        resp = requests.get(url, timeout=30)
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        headers = [th.get_text(strip=True) for th in table.find_all("th")]

        rows = []
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            # pad / trim to header length
            if len(cells) < len(headers):
                cells += [""] * (len(headers) - len(cells))
            elif len(cells) > len(headers):
                cells = cells[: len(headers)]
            rows.append(cells)

        df = pd.DataFrame(rows, columns=headers)

    # 🔧 Sanitise column names for SQL
    df.columns = [_sql_safe(c) for c in df.columns]

    # Build list-version with the same headers
    list_version = [df.columns.tolist()] + df.values.tolist()

    return [df, list_version]

# ---------------------------------------------------------------------
# 2. DuckDB helper
# ---------------------------------------------------------------------
def run_duckdb_query(query: str, files: dict | None = None):
    """
    Execute SQL against DuckDB.
      files = {"view_name": DataFrame_or_path, …}
    Accepts pandas DataFrames as well as file paths.
    """
    con = duckdb.connect()
    try:
        if files:
            for name, obj in files.items():
                if isinstance(obj, pd.DataFrame):
                    con.register(name, obj)
                else:  # assume path string
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
    Accept a matplotlib Figure *or* the plt module; return data-URI.
    """
    if not hasattr(fig, "savefig"):      # user passed plt
        fig = fig.gcf()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    data_uri = "data:image/png;base64," + base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data_uri
