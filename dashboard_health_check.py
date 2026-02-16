#!/usr/bin/env python3
import os, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from db import connect, init_db

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
DB_PATH = os.getenv('BLOFIN_DB_PATH', str(ROOT / 'data' / 'blofin_monitor.db'))

TARGETS = [
    ('blofin', 'http://127.0.0.1:8780/healthz'),
]


def now_pair():
    t = int(time.time() * 1000)
    iso = datetime.fromtimestamp(t/1000, tz=timezone.utc).isoformat()
    return t, iso


def check(url):
    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            code = r.status
            ok = 1 if 200 <= code < 300 else 0
            err = ''
    except Exception as e:
        code = 0
        ok = 0
        err = str(e)
    lat = int((time.time() - start) * 1000)
    return ok, code, lat, err


def main():
    con = connect(DB_PATH)
    init_db(con)
    ts_ms, ts_iso = now_pair()
    for name, url in TARGETS:
        ok, code, lat, err = check(url)
        con.execute('INSERT INTO dashboard_checks(ts_ms,ts_iso,name,url,ok,status_code,latency_ms,error) VALUES(?,?,?,?,?,?,?,?)',
                    (ts_ms, ts_iso, name, url, ok, code, lat, err))
    con.commit()


if __name__ == '__main__':
    main()
