"""Restore the exact source images listed in the supplied manifest."""
import hashlib
import json
from pathlib import Path
import urllib.request

def main():
    manifest=json.loads(Path('data/manifest.json').read_text())
    base=f'https://raw.githubusercontent.com/openalpr/benchmarks/{manifest["commit"]}/'
    for record in manifest['images']:
        path=Path(record['image'])
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']:
            continue
        with urllib.request.urlopen(base+record['source'],timeout=90) as response:
            content=response.read()
        if hashlib.sha256(content).hexdigest()!=record['sha256']:
            raise ValueError(f'Checksum mismatch: {record["id"]}')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(content)
        print(record['id'])

if __name__=='__main__': main()
