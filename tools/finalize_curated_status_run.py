#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_curated_status_lstm as t

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run', default='runs/curated_status_h320_l3')
    args=ap.parse_args()
    out=Path(args.run)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ckpt=torch.load(out/'best.pt', map_location=device)
    a=ckpt['args']
    model=t.CharLSTM(a['hidden'], a['layers'], dropout=0.04).to(device)
    model.load_state_dict(ckpt['model_state'])
    splits=t.build_curated_corpus(a['repeats'], a['seed'])
    val=t.encode(splits.val); test=t.encode(splits.test)
    val_loss=t.eval_loss(model, val, device, a['seq_len'])
    test_loss=t.eval_loss(model, test, device, a['seq_len'])
    policy=t.eval_policy_continuations(model, device, length=48)
    exports=[]
    for profile in ['all_int8','mixed_lstm_safe']:
        name=f"curated_status_h{a['hidden']}_l{a['layers']}_{profile}.bin"
        exports.append(t.export_rilm(model, out/'weights'/name, profile))
    samples={prompt:t.generate(model,prompt,80,device,'greedy') for prompt,_ in t.POLICY_CASES}
    summary={
        'schema':'ri_esp32_curated_status_lstm_train_v1',
        'finalized_from':'best.pt_after_thermal_stop',
        'hidden':a['hidden'], 'layers':a['layers'], 'params':model.count_params(),
        'device':device, 'cuda_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'corpus_manifest':splits.manifest,
        'best_val_loss':float(ckpt['best_val_loss']), 'best_val_ppl':math.exp(float(ckpt['best_val_loss'])),
        'finalized_val_loss':val_loss, 'finalized_val_ppl':math.exp(val_loss),
        'finalized_test_loss':test_loss, 'finalized_test_ppl':math.exp(test_loss),
        'policy_eval':policy, 'exports':exports, 'samples':samples,
        'claim_boundary':'MSI/GTX-trained and RILM-exported. Not an ESP32 hardware claim until flashed and BENCH_RECEIPT captured.'
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps({'summary':str(out/'summary.json'),'params':summary['params'],'best_val_ppl':summary['best_val_ppl'],'test_ppl':summary['finalized_test_ppl'],'prefix_acc':policy['prefix_accuracy'],'contains_acc':policy['contains_accuracy'],'exports':exports},indent=2,sort_keys=True))
if __name__=='__main__': main()
