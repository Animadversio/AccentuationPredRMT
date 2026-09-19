"""Build a content-audited 25-site x 10-model geometry manifest."""
import argparse
import csv
import hashlib
import json
import pickle
import re
from pathlib import Path

import yaml


CONFIG_ROOT = Path('/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_configs')
CACHE_ROOT = Path('/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models')


def short_hash(data):
    return hashlib.sha256(data).hexdigest()[:16]


def safe_name(value):
    return re.sub(r'[^A-Za-z0-9._-]+', '-', value).strip('-')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for config_path in sorted(CONFIG_ROOT.glob('*/*.yaml')):
        config = yaml.safe_load(config_path.read_text())
        subject = config['subject_id']
        unit = int(config['unit_ids'][0])
        model = config['model_name']
        cache_path = (CACHE_ROOT/subject/'posthoc_model_predict_PCA_popul_unit'/
                      f'posthoc_prediction_NSDencimg_PCA_pop_unit_{subject}_unit{unit}_{model}.pkl')
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        with cache_path.open('rb') as file:
            cache = pickle.load(file)
        scores = cache['PCA_resp'].detach().cpu().numpy()
        frame = cache['df']
        split_payload = ('\n'.join(frame.image_fps.astype(str))+'\n'+
                         ''.join(frame.is_train.astype(int).astype(str))).encode()
        score_hash = short_hash(memoryview(scores))
        split_hash = short_hash(split_payload)
        seed_hash = short_hash('\n'.join(config['seed_image_paths']).encode())
        basis_key = f'{model}\n{config["layer_name"]}\n{score_hash}\n{split_hash}\n{seed_hash}'.encode()
        geometry_id = f'{safe_name(model)}__{safe_name(config["layer_name"])}__{short_hash(basis_key)[:10]}'
        rows.append(dict(
            geometry_id=geometry_id, subject=subject, unit=unit, model=model,
            layer=config['layer_name'], score_hash=score_hash, split_hash=split_hash,
            seed_hash=seed_hash, n_total=scores.shape[0], n_pcs=scores.shape[1],
            n_train=int(frame.is_train.sum()), config_path=str(config_path),
            cache_path=str(cache_path), xtransform_path=config['xtransform_path'],
            readout_path=config['readout_path']))
    if len(rows) != 250:
        raise ValueError(f'Expected 250 rows, found {len(rows)}')
    if len({r['split_hash'] for r in rows}) != 1 or len({r['seed_hash'] for r in rows}) != 1:
        raise ValueError('Evaluation images or training split differ across rows')
    groups = {}
    for row in rows:
        groups.setdefault(row['geometry_id'], []).append(row)
    # Exact score equality is a strong audit of exported PCA coordinates. The
    # production runner should additionally compare representative PCA buffers
    # for duplicate rows before using aliases.
    geometry_rows = []
    for geometry_id, members in sorted(groups.items()):
        rep = sorted(members, key=lambda r: (r['subject'], r['unit']))[0]
        geometry_rows.append(dict(
            geometry_id=geometry_id, model=rep['model'], layer=rep['layer'],
            representative_subject=rep['subject'], representative_unit=rep['unit'],
            representative_config=rep['config_path'], representative_cache=rep['cache_path'],
            representative_xtransform=rep['xtransform_path'], score_hash=rep['score_hash'],
            split_hash=rep['split_hash'], seed_hash=rep['seed_hash'], n_total=rep['n_total'],
            n_train=rep['n_train'], n_pcs=rep['n_pcs'], alias_count=len(members),
            site_keys=';'.join(f'{r["subject"]}:unit{r["unit"]}' for r in sorted(members,key=lambda x:(x['subject'],x['unit'])))))
    for name, values in [('site_model_manifest.csv', rows), ('geometry_manifest.csv', geometry_rows)]:
        with (args.output/name).open('w', newline='') as file:
            writer = csv.DictWriter(
                file, fieldnames=list(values[0]), lineterminator='\n'
            )
            writer.writeheader(); writer.writerows(values)
    per_model = {}
    for model in sorted({r['model'] for r in rows}):
        model_geometries = [r for r in geometry_rows if r['model']==model]
        per_model[model] = dict(pairs=sum(r['model']==model for r in rows),
                                unique_geometries=len(model_geometries),
                                layers=[r['layer'] for r in model_geometries])
    summary = dict(site_model_pairs=len(rows), sites=len({(r['subject'],r['unit']) for r in rows}),
                   subjects=len({r['subject'] for r in rows}), models=len({r['model'] for r in rows}),
                   unique_geometries=len(geometry_rows), unique_split_hashes=len({r['split_hash'] for r in rows}),
                   unique_seed_hashes=len({r['seed_hash'] for r in rows}), per_model=per_model,
                   audit_note='Geometry aliases have exact PCA-score hashes; verify exported PCA buffers before production deduplication.')
    (args.output/'inventory_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
