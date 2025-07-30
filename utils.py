import requests
from bs4 import BeautifulSoup
import duckdb
import matplotlib.pyplot as plt
import base64
import io


def scrape_table_from_url(url):
    """
    Scrape the first HTML table from the given URL and return as a list of dicts (rows).
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')
    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    rows = []
    for tr in table.find_all('tr')[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return rows


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
