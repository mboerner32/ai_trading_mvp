import requests
from bs4 import BeautifulSoup


FINVIZ_URL = (
    "https://finviz.com/screener.ashx?"
    "v=111&f=geo_usa,sh_curvol_o5000,sh_price_u5,sh_relvol_o10,ta_perf_d10o&ft=4"
)


def get_finviz_universe():
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(FINVIZ_URL, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    tickers = []

    # Tickers are in links like quote.ashx?t=ATOM
    for link in soup.find_all("a", href=True):
        href = link["href"]

        if "quote.ashx?t=" in href:
            ticker = href.split("t=")[1].split("&")[0]

            if ticker.isupper() and len(ticker) <= 5:
                tickers.append(ticker)

    # Remove duplicates
    tickers = list(set(tickers))

    print("FOUND TICKERS:", tickers)
    return tickers