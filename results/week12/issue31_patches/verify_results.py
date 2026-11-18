"""Verify compact issue-31 records; --raw also checks local simulation artifacts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import torch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--raw', action='store_true', help='Also verify local, untracked simulation artifacts')
args=parser.parse_args()
output=Path(__file__).resolve().parent
root=output.parents[2]
def load(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
archive_root=Path(load(output/'execution.json')['cwd'])
def local_path(value):
 path=Path(value)
 relative=path.relative_to(archive_root)
 assert '..' not in relative.parts
 return root/relative
def close(actual,expected): assert np.isclose(actual,expected,rtol=1e-10,atol=1e-12),(actual,expected)
config=load(output/'default_config.json'); settings=load(output/'search_settings.json'); selected=load(output/'selected_config.json'); opt=load(output/'optimizer.json')
assert config==load(root/'configs/coralai_supervised.default.json')
assert settings==load(root/'configs/coralai_supervised.search.json')
assert settings['search_seeds']==[42,7,13,21,99]
assert settings['comparison_seeds']==[101,202,303]
assert not set(settings['search_seeds']) & set(settings['comparison_seeds'])
assert config['steps']==1000 and config['shape']==100 and config['env']=='patches' and config['env_param']==8
assert opt['population']==6 and opt['generations']==5 and opt['optimizer_seed']==0 and opt['sigma']==0.1
assert opt['spread_penalty']==settings['spread_penalty']==1.0
assert {k:v for k,v in selected.items() if k not in settings['parameters']}=={k:v for k,v in config.items() if k not in settings['parameters']}
candidates=load(output/'candidates.json'); generations=load(output/'generations.json'); completion=load(output/'completed.json'); selection=load(output/'selection.json')
expected_ids={f'g{g:02d}_c{c:02d}' for g in range(5) for c in range(6)}
assert len(candidates)==30 and {c['id'] for c in candidates}==expected_ids
assert len(generations)==5 and [g['generation'] for g in generations]==list(range(5))
assert completion['evaluations']==155
expected_steps=list(range(50,1001,50)); counts={'evaluations':0,'scored_frames':0,'saved_frames':0,'finite_final_states':0}
def evaluation(folder,expected_config,seed,score=None):
 if score is None: score=load(folder/'scores.json')
 if args.raw: assert score==load(folder/'scores.json')
 run=folder/'run'
 if args.raw or (folder/'input_config.json').exists():
  assert load(folder/'input_config.json')==dict(expected_config,seed=seed,output_dir=None,resume_from=None)
 if args.raw:
  runtime=load(run/'config.json'); meta=load(run/'meta.json')
  assert meta['effective_config']==runtime
  assert local_path(runtime['output_dir'])==run
  assert runtime==dict(expected_config,seed=seed,output_dir=runtime['output_dir'],resume_from=None)
  assert meta['completed_steps']==1000 and meta['start_step']==0 and meta['seed']==seed and meta['shape']==[100,100]
  assert meta['final_checkpoint']=='checkpoint_0001000'
  assert meta['env']['name']=='patches' and meta['env']['n_patches']==8
 if args.raw or (folder/'command.json').exists():
  command=load(folder/'command.json')
  assert Path(command['cwd'])==archive_root
  assert local_path(command['argv'][1])==root/'external/coralai/experiments/coral_dev/headless_coral.py'
  assert local_path(command['PYTHONPATH'].split(':')[0])==root/'external/coralai'
 assert score['seed']==seed
 assert local_path(score['final_checkpoint'])==run/'checkpoint_0001000'
 frames=score['frame_scores']; assert [f['step'] for f in frames]==expected_steps
 losses=np.asarray([f['loss'] for f in frames],dtype=np.float32); assert np.isfinite(losses).all()
 close(score['mean_loss'],losses.mean()); close(score['mean_alignment'],-losses.mean())
 close(score['best_sample_alignment'],-losses.min()); close(score['final_alignment'],-losses[-1])
 assert score['best_sample_step']==expected_steps[int(losses.argmin())]
 assert score['simulation_seconds']>0 and score['scoring_seconds']>0
 for f in frames:
  assert local_path(f['image'])==run/'screenshots'/f"step_{f['step']:07d}.png"
  assert len(f['sha256'])==64
  close(f['alignment'],-f['loss'])
 if (folder/'final.png').exists():
  assert digest(folder/'final.png')==frames[-1]['sha256']
  with Image.open(folder/'final.png') as image:
   assert image.size==(100,100) and image.mode=='RGB'; image.verify()
 if args.raw:
  saved=sorted((run/'screenshots').glob('*.png'))
  assert [p.name for p in saved]==[f'step_{s:07d}.png' for s in [0,*expected_steps]]
  for path in saved:
   with Image.open(path) as image:
    assert image.size==(100,100) and image.mode=='RGB'; image.verify()
  for f in frames: assert f['sha256']==digest(local_path(f['image']))
  assert digest(folder/'best_sample.png')==frames[int(losses.argmin())]['sha256']
  assert digest(folder/'final.png')==frames[-1]['sha256']
  checkpoint=run/'checkpoint_0001000'; assert load(checkpoint/'meta.json')['step']==1000
  assert (checkpoint/'population.pkl').stat().st_size>0
  state=torch.load(checkpoint/'substrate.pt',map_location='cpu',weights_only=True)
  assert tuple(state.shape)==(1,14,100,100) and bool(torch.isfinite(state).all())
  with (run/'step_log.csv').open() as stream: rows=list(csv.DictReader(stream))
  assert len(rows)==100 and [int(row['step']) for row in rows]==list(range(0,1000,10))
  assert all(np.isfinite(float(value)) for row in rows for value in row.values())
  counts['saved_frames']+=len(saved); counts['finite_final_states']+=1
 counts['evaluations']+=1; counts['scored_frames']+=len(frames)
 return score

def multi(folder,expected_config,summary=None):
 if summary is None: summary=load(folder/'multi_seed_summary.json')
 if args.raw: assert summary==load(folder/'multi_seed_summary.json')
 assert len(summary['per_seed'])==5
 scores=[evaluation(folder/f'seed_{seed}',expected_config,seed,saved) for seed,saved in zip(settings['search_seeds'],summary['per_seed'])]
 assert summary['per_seed']==scores and summary['seeds']==settings['search_seeds'] and summary['n_seeds']==5
 losses=np.array([s['mean_loss'] for s in scores]); close(summary['mean_loss'],losses.mean()); close(summary['loss_std'],losses.std(ddof=1)); close(summary['selection_loss'],losses.mean()+losses.std(ddof=1))
 close(summary['mean_alignment'],-losses.mean()); close(summary['alignment_std'],losses.std(ddof=1))
 return summary
baseline=multi(output/'default',config); assert baseline==selection['default_scores']
all_scores=[('default',baseline,config)]
for c in candidates:
 for x,(name,spec) in zip(c['vector'],settings['parameters'].items()):
  raw=spec['min']+float(np.clip(x,0,1))*(spec['max']-spec['min'])
  assert c['parameters'][name]==(int(round(raw)) if spec['integer'] else raw)
 candidate_config=dict(config,**c['parameters'])
 score=multi(output/'candidates'/c['id'],candidate_config,c['scores'])
 assert c['scores']==score
 all_scores.append((c['id'],score,candidate_config))
best=min(all_scores,key=lambda item:item[1]['selection_loss'])
assert selection['selected_id']==completion['selected_id']==best[0]
assert selection['selected_scores']==best[1] and selected==dict(best[2],output_dir=None,resume_from=None)
for g in generations:
 assert g['losses']==[c['scores']['selection_loss'] for c in candidates if c['generation']==g['generation']]
 assert (output/f"optimizer_state_{g['generation']:02d}.npz").is_file()
assert counts['evaluations']==155
heldout=output/'heldout_check'; per_seed=load(heldout/'per_seed.json'); summary=load(heldout/'summary.json')
assert [r['seed'] for r in per_seed]==settings['comparison_seeds']
for row in per_seed:
 for label,cfg in [('default',config),('selected',selected)]:
  assert row[label]==evaluation(heldout/f"seed_{row['seed']}"/label,cfg,row['seed'])
 close(row['difference'],row['selected']['mean_alignment']-row['default']['mean_alignment'])
differences=np.array([r['difference'] for r in per_seed])
assert summary['n_seeds']==3 and summary['improved_seeds']==int((differences>0).sum())
close(summary['paired_difference_mean'],differences.mean()); close(summary['paired_difference_sample_std'],differences.std(ddof=1))
for label in ('default','selected'):
 values=np.array([r[label]['mean_alignment'] for r in per_seed])
 close(summary[label+'_mean_alignment'],values.mean()); close(summary[label+'_sample_std'],values.std(ddof=1))
assert counts['evaluations']==161 and counts['scored_frames']==3220
if args.raw: assert counts['saved_frames']==3381 and counts['finite_final_states']==161
recorded_validation=load(output/'validation.json')
assert recorded_validation['status']=='passed' and recorded_validation['finite_final_states']==161 and recorded_validation['saved_frames']==3381
assert summary['improved_seeds']==1
close(summary['paired_difference_mean'],-0.0025497525930404663)
for folder in (output,heldout):
 provenance=load(folder/'execution.json')
 assert provenance['git_head']=='e610d890ef96d5a72f83ab2c84a3ca8f6112b7b4'
 for rel,sha in provenance['source_sha256'].items(): assert digest(root/rel)==sha
 scorer=load(folder/'scorer.json')
 assert scorer['revision']==settings['revision'] and scorer['device']=='TFRT_CPU_0'
 assert scorer['weights_sha256']=='2a2994d89bebd77abba5a554789dd9152a7e25467b79d88f6bd237d2dec5051c'
subprocess.run(['git','apply','--reverse','--check',str(output/'controller.patch')],cwd=root,check=True)
report={'status':'passed','mode':'raw' if args.raw else 'compact','python_version':sys.version,'evaluations_checked':counts['evaluations'],'frame_scores_checked':counts['scored_frames'],'raw_saved_frames_checked':counts['saved_frames'],'raw_final_states_checked':counts['finite_final_states'],'original_full_validation':recorded_validation['status'],'selected_id':best[0],'selected_parameters':{k:selected[k] for k in settings['parameters']},'source_hashes_match':True,'controller_patch_matches_source':True,'heldout_summary':summary,'note':'Historical absolute paths are mapped from the recorded repository root to this checkout. Compact mode rechecks scores/selection and the six held-out final PNGs; validation.json preserves the completed original full raw-artifact validation. No simulations are launched or records overwritten.'}
assert len(list(output.glob('heldout_check/seed_*/*/final.png')))==6
print(json.dumps(report,indent=2))
