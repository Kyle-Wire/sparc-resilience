"""Quick check of artifact DB contents."""
import sqlite3, sys, json
from pathlib import Path

cities = ["raleigh_nc", "chicago_il", "atlanta_ga", "seattle_wa"]
for city in cities:
    db_path = Path(f"output/cities/{city}/output/artifacts.db")
    if not db_path.exists():
        print(f"{city}: DB not found at {db_path}")
        continue
    conn = sqlite3.connect(str(db_path))
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print(f"\n{city}: tables={[t[0] for t in tables]}")
    for (tname,) in tables:
        cnt = conn.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info({tname})").fetchall()]
        print(f"  {tname}: {cnt} rows, cols={cols}")
        if cnt > 0:
            rows = conn.execute(f"SELECT * FROM {tname} LIMIT 3").fetchall()
            for r in rows:
                print(f"    {r[:6]}")
    conn.close()
