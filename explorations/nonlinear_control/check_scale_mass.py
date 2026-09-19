"""Validate scale-up completion and write a machine-readable status report."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    geometry_root = args.output / 'geometry'
    rows = []
    for row in manifest.itertuples():
        directory = geometry_root / row.geometry_id
        done_path = directory / 'DONE.json'
        seed_count = len(list(directory.glob('seed_??.npz'))) if directory.exists() else 0
        state = 'complete' if done_path.exists() and seed_count == 10 else 'incomplete'
        rows.append(dict(
            geometry_id=row.geometry_id, model=row.model, layer=row.layer,
            state=state, seed_archives=seed_count, done_json=done_path.exists(),
        ))
    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / 'status.csv', index=False)
    summary = dict(
        expected=len(frame), complete=int((frame.state == 'complete').sum()),
        incomplete=int((frame.state != 'complete').sum()),
        per_model=frame.groupby(['model', 'state']).size().unstack(fill_value=0).to_dict('index'),
    )
    (args.output / 'status.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if summary['incomplete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
