"""Validate and extract the public .iml archive without accessing Lustre.

Source: https://pirsquared.org/research/vhatdb/full/vanhateren_iml.zip
The expected filenames below were checked against the original experiment's
4167-file dataset directory on 2026-09-17 (names only, no image reads).
"""
import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from tqdm.auto import tqdm

MISSING = {2829,2831,2835,2837,2839,2846,2859,2860,2863,2864,2865,2872,
           2873,2875,2876,2877,2879,2880,2882,2885,2886,2888,2890,2891,
           2892,2893,2922,2924,2925,2927,2936,2938,2939,2941,2963,2964,
           2965,2969,2971,2974,2975,2977,2984,3000,3001}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    expected={f'imk{i:05d}.iml' for i in range(1,4213) if i not in MISSING}
    args.output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    with zipfile.ZipFile(args.archive) as z:
        members=[m for m in z.infolist() if not m.is_dir() and m.filename.endswith('.iml')]
        names=[Path(m.filename).name for m in members]
        if set(names)!=expected or len(names)!=len(expected):
            raise ValueError('Archive does not match original image universe')
        if any(m.file_size!=1024*1536*2 for m in members):
            raise ValueError('Unexpected raw image dimensions/byte count')
        manifest=[]
        for i,m in enumerate(tqdm(members,desc='Extract/CRC-check original .iml images')):
            name=Path(m.filename).name
            destination=args.output/name
            partial=destination.with_suffix('.iml.partial')
            # Only validated basenames are used; ZIP paths are never extracted.
            with z.open(m) as src, partial.open('wb') as out:
                shutil.copyfileobj(src,out,length=1024*1024)
            partial.replace(destination)
            manifest.append({'name':name,'size':m.file_size,'crc32':m.CRC})
            if i==9:
                print(f'Extraction pilot: 10 files in {time.perf_counter()-start:.2f}s; '
                      f'ETA {(time.perf_counter()-start)*len(members)/10:.1f}s',flush=True)
    digest=hashlib.sha256()
    with args.archive.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    (args.output/'download_manifest.json').write_text(json.dumps({
        'source':'https://pirsquared.org/research/vhatdb/full/vanhateren_iml.zip',
        'archive_sha256':digest.hexdigest(),'image_count':len(manifest),
        'files':manifest},indent=2))
    print(f'Validated {len(manifest)} original filenames, sizes and ZIP CRCs; '
          f'elapsed {time.perf_counter()-start:.1f}s; SHA256 {digest.hexdigest()}',flush=True)


if __name__=='__main__':main()
