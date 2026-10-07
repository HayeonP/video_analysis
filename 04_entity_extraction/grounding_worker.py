#!/usr/bin/env python3
"""Optional GPU worker for already-prepared samples; writes independent frame caches."""
import argparse
import logging
from pathlib import Path
import os
from extract_entities import Settings, QwenBackend, read_json, atomic_json, detect_batch, Image, torch


def main():
    p=argparse.ArgumentParser(description='Additional grounding worker; start after the main process prepares proposals.')
    p.add_argument('output',type=Path)
    p.add_argument('--start',type=int,required=True)
    p.add_argument('--end',type=int,required=True)
    p.add_argument('--reverse',action='store_true')
    args=p.parse_args();output=args.output.resolve()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    torch.set_num_threads(4)
    cfg=read_json(output/'run_config.json');settings=Settings(**cfg['settings'])
    frames=read_json(output/'samples.json');proposals=read_json(output/'yolo_proposals.json')
    backend=QwenBackend(settings,output/'workers'/os.environ.get('CUDA_VISIBLE_DEVICES','default')/'traces')
    size=cfg['batch_size']
    if args.start%size or args.end>len(frames): raise ValueError('Use aligned start and valid end')
    offsets=list(range(args.start,args.end,size))
    if args.reverse: offsets.reverse()
    for offset in offsets:
        batch=[f for f in frames[offset:min(offset+size,args.end)] if not (output/'grounding'/f'{f["frame_key"]}.json').exists()]
        if not batch: continue
        logging.info('Worker grounding %d-%d',offset+1,min(offset+size,args.end))
        images=[Image.open(output/f['image']).convert('RGB') for f in batch]
        for result in detect_batch(backend,batch,images,proposals):
            # Main process may have reached this batch too; retain its first completed result.
            destination=output/'grounding'/f'{result["frame_key"]}.json'
            if not destination.exists(): atomic_json(destination,result)


if __name__=='__main__': main()
