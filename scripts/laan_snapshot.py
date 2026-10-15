"""Gemmer dagens kurser på Totalkredit 5 % 2059 og 4 % 2059 i laan/history.json.

Køres af GitHub Actions efter børsens lukketid. Data er offentlige kurser; intet personligt.
"""
import json
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

HIST = Path(__file__).resolve().parent.parent / "laan" / "history.json"
TK = "https://www.totalkredit.dk/api/bondinformation/table?tableId=privat-udbetaling-af-laan-aktuelle-kurser-kunder&domain=totalkredit"
NASDAQ = "https://api.nasdaq.com/api/nordic/instruments/{id}/chart?assetClass=MORTGAGE_BONDS&fromDate={fra}&toDate={til}&lang=en"
BONDS = {"k5": ("955620", "TX7645240"), "k4": ("955299", "TX6851322")}
UA = {"User-Agent": "Mozilla/5.0 (Macintosh) laan-snapshot"}


def hent(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)


def tal(s):
    try:
        return float(str(s).replace(".", "").replace(",", ".").replace("%", "").strip())
    except ValueError:
        return None


def main():
    i_dag = datetime.now(ZoneInfo("Europe/Copenhagen")).date()
    if i_dag.weekday() >= 5:
        return
    rows = json.loads(HIST.read_text()) if HIST.exists() else []
    row = next((r for r in rows if r["date"] == i_dag.isoformat()), None)
    if row is None:
        row = {"date": i_dag.isoformat()}
        rows.append(row)

    try:
        tk = hent(TK)
        entries = {e["fondCode"]: e for g in tk["groups"] for e in g["entries"]}
        for k, (fond, _) in BONDS.items():
            e = entries.get(fond)
            if e:
                row[f"{k}_tilbud"] = tal(e["priceRate"])
                row[f"{k}_aktuel"] = tal(e["spotPriceRatePayment"])
                row[f"{k}_aaben"] = bool(e["isOpenForOffer"])
    except Exception as ex:  # Totalkredit nede: behold resten
        print("Totalkredit fejlede:", ex)

    fra = (i_dag - timedelta(days=10)).isoformat()
    for k, (_, nid) in BONDS.items():
        try:
            cp = hent(NASDAQ.format(id=nid, fra=fra, til=i_dag.isoformat()))["data"]["CP"]
            for p in cp:
                d = p["z"]["dateTime"]
                r = next((x for x in rows if x["date"] == d), None)
                if r is None:
                    r = {"date": d}
                    rows.append(r)
                r[f"{k}_nasdaq"] = float(p["z"]["close"])
        except Exception as ex:  # Nasdaq blokerer ofte cloud-IP'er
            print("Nasdaq fejlede for", k, ex)

    rows.sort(key=lambda r: r["date"])
    HIST.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
