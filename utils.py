import requests
from bs4 import BeautifulSoup
import duckdb
import matplotlib.pyplot as plt
import base64
import io
import pandas as pd

def scrape_table_from_url(url: str):
    """
    Return a list with two elements:
      tables[0] = pandas DataFrame (most robust for analysis)
      tables[1] = list-of-lists  [ header_row, data_row1, ... ]
    This lets either access style work.
    """
    try:
        df = pd.read_html(url, flavor="bs4")[0]
    except Exception:
        # fallback scrape
        resp = requests.get(url, timeout=30)
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")

        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        rows = [
            [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            for tr in table.find_all("tr")[1:]
        ]
        df = pd.DataFrame(rows, columns=headers)

    # Build list-of-lists version
    list_version = [df.columns.tolist()] + df.values.tolist()

    return [df, list_version]

def run_duckdb_query(query, files=None):
    """
    Run a DuckDB SQL query. Optionally register files (e.g., parquet) for querying.
    """
    con = duckdb.connect()
    if files:
        for name, path in files.items():
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
    result = con.execute(query).fetchall()
    columns = [desc[0] for desc in con.description]
    return [dict(zip(columns, row)) for row in result]


def plot_and_encode_base64(fig):
    """
    Encode a matplotlib figure as a base64 PNG data URI.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_bytes = buf.read()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    data_uri = f"data:image/png;base64,{base64_str}"
    plt.close(fig)
    return data_uri 
