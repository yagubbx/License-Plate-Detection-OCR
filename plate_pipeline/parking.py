import json
import sqlite3
from datetime import datetime, timezone
import time

class ParkingLog:
    def __init__(self,path,cooldown=30):
        self.db=sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS sightings (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, source TEXT NOT NULL, plate TEXT NOT NULL, box TEXT NOT NULL, confidence REAL NOT NULL)')
        self.last={}; self.cooldown=cooldown

    def record(self,predictions,source):
        now=time.monotonic()
        for p in predictions:
            if not p['text'] or p.get('format_valid') is False:
                continue
            key=(str(source),p['text'])
            if now-self.last.get(key,float('-inf')) < self.cooldown:
                continue
            self.db.execute('INSERT INTO sightings(timestamp,source,plate,box,confidence) VALUES(?,?,?,?,?)',
                            (datetime.now(timezone.utc).isoformat(),str(source),p['text'],json.dumps(p['box']),p['confidence']))
            self.last[key]=now
        self.db.commit()

    def close(self):
        self.db.close()
