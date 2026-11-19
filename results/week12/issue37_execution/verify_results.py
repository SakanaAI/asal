from pathlib import Path
import ast,csv,hashlib,json,math,sys
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, str(Path.cwd()))

from scripts.temporal_robustness import aggregate_losses,paired_heldout,change_summary,TIE
from scripts.coralai_search import decode
from substrates.coralai import render_checkpoint

ROOT=Path.cwd()
OUTPUT=Path(sys.argv[1]).resolve();BUDGET=int(sys.argv[2])
SETTINGS=json.loads((OUTPUT/'search_settings.json').read_text());CONFIG=json.loads((OUTPUT/'base_config.json').read_text())
SEEDS=SETTINGS['search_seeds'];UPDATES=SETTINGS['sampled_updates'];PENALTY=SETTINGS['spread_penalty']

def load(path):return json.loads(path.read_text())
def need(ok,message):
 if not ok:raise ValueError(message)
def close(a,b,where,tol=2e-7):need(math.isfinite(float(a)) and abs(float(a)-float(b))<=tol,f'{where}: {a} != {b}')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

need(SEEDS==[303,7,13] and SETTINGS['comparison_seeds']==[101,202,404],'Wrong seed plan')
completed=load(OUTPUT/'completed.json');need(completed['candidate_worlds']==BUDGET and completed['rollouts_per_world']==3,'Incomplete candidate budget')
candidates=load(OUTPUT/'candidates.json');need(len(candidates)==BUDGET,'Missing candidates')
need(len({r['id'] for r in candidates})==BUDGET,'Duplicate candidate IDs')
models=set();fingerprints=set();rollouts=[];initials={};groups={}

def check_run(directory,seed,parameters=None):
 score=load(directory/'scores.json');effective=load(directory/'effective_config.json')
 contract=load(directory/'reference_contract.json');run=Path(score['run'])
 need(run.is_relative_to(OUTPUT) and run.name=='run',f'Unrelated raw run {run}')
 need(effective['seed']==seed and effective['steps']==200 and effective['shape']==48,'Wrong run seed/horizon/grid')
 for key in ('env','env_param','infra_decay','defense_coeff','radiate_interval','cleanup_interval','emergency_pop_cap','backend','device'):
  need(effective[key]==CONFIG[key],f'Changed scientific setting {key}')
 need(contract['sampled_updates']==UPDATES and not contract['initial_update_scored'],'Wrong scored update plan')
 need(contract['scoring']['view']=='occupancy' and contract['scoring']['model']==SETTINGS['model'] and contract['scoring']['revision']==SETTINGS['revision'],'Wrong scoring/render/model')
 models.add(contract['scoring']['weights_sha256']);fingerprints.add(json.dumps(contract['simulator'],sort_keys=True))
 if parameters is not None:
  expected=dict(CONFIG['env_options']);expected.update(parameters)
  need(effective['env_options']==expected,'Effective parameters differ from proposal')
 embeddings=np.load(directory/'embeddings.npy')
 need(embeddings.shape==(10,512) and np.isfinite(embeddings).all(),'Bad embeddings')
 need(np.allclose(np.linalg.norm(embeddings,axis=1),1,atol=1e-4),'Unnormalized embeddings')
 loss=float(np.tril(embeddings@embeddings.T,k=-1).max(axis=1).mean())
 close(score['loss_oe'],loss,str(directory/'scores.json'),tol=3e-7)
 close(score['novelty'],.9-score['loss_oe'],'Novelty display',tol=1e-12)
 need(len(score['measurements'])==11,'Incomplete measurements')
 for u,row in zip([0,*UPDATES],score['measurements']):need(row['completed_update']==u,'Mismatched completed updates')
 first=torch.load(run/'frame_states/update_0000000.pt',weights_only=True)
 if seed in initials:need(torch.equal(first,initials[seed]),f'Initial state differs within seed {seed}')
 else:initials[seed]=first
 for u in (0,100,200):
  expected=render_checkpoint(run/'frame_states'/f'update_{u:07d}.pt',view='occupancy')
  with Image.open(directory/'scoring_images'/f'update_{u:07d}.png') as im:actual=np.array(im.convert('RGB'),dtype=np.float32)/255
  need(np.array_equal(expected,actual),'Scoring image differs from its saved state')
 rollouts.append({'directory':str(directory.relative_to(OUTPUT)),'seed':seed,'loss':score['loss_oe'],'simulation_seconds':score['simulation_seconds'],'scoring_seconds':score['scoring_seconds'],'reused':score['reused']})
 return score,effective


def check_world(directory,parameters=None):
 scores=[]
 for i,seed in enumerate(SEEDS):
  where=directory if i==0 else directory/f'seed_{seed}'
  score,_=check_run(where,seed,parameters)
  scores.append(score)
 expected=aggregate_losses([s['loss_oe'] for s in scores],PENALTY)
 summary=load(directory/'multi_seed_summary.json')
 need(summary['seeds']==SEEDS,'World missing a search seed')
 for field in ('mean_loss','loss_std','selection_loss'):close(summary[field],expected[field],field,tol=1e-12)
 return expected

fixed,_=check_run(OUTPUT/'fixed',303,{'shift_interval':0})
hand=check_world(OUTPUT/'hand',CONFIG['env_options'])
best,best_id=hand,'hand'
for candidate in candidates:
 parameters=decode(candidate['vector'],SETTINGS['parameters'])
 need(all(candidate[k]==v for k,v in parameters.items()),'Parameter rounding differs')
 groups.setdefault(tuple(parameters.items()),[]).append(candidate['id'])
 summary=check_world(OUTPUT/'candidates'/candidate['id'],parameters)
 for field in ('mean_loss','loss_std','selection_loss'):close(candidate[field],summary[field],field,tol=1e-12)
 need(np.array_equal(candidate['per_seed_loss'],summary['per_seed_loss']),'Per-seed candidate values differ')
 if summary['selection_loss']<best['selection_loss']-TIE:best,best_id=summary,candidate['id']
selection=load(OUTPUT/'selection.json');need(selection['selected_id']==best_id,'Wrong selected candidate')
for field in ('mean_loss','loss_std','selection_loss'):close(selection['selected'][field],best[field],field,tol=1e-12)
close(selection['selection_loss_improvement_vs_hand'],hand['selection_loss']-best['selection_loss'],'Selection improvement',tol=1e-12)
with (OUTPUT/'candidates.csv').open() as stream:
 csv_rows=list(csv.DictReader(stream))
need(len(csv_rows)==BUDGET,'Incomplete candidate CSV')
for row,record in zip(csv_rows,candidates):
 need(row['id']==record['id'],'CSV candidate identity differs')
 for field in ('selection_loss','mean_loss','loss_std'):close(row[field],record[field],field,tol=1e-12)
 need(ast.literal_eval(row['per_seed_loss'])==record['per_seed_loss'],'CSV per-seed values differ')

heldout=None
if completed['heldout_ran']:
 expected_losses={'hand':[],'selected':[]};native_rows=load(OUTPUT/'heldout/per_seed.json')
 selected_config=load(OUTPUT/'selected_config.json')
 for seed,native in zip(SETTINGS['comparison_seeds'],native_rows):
  need(seed==native['seed'],'Wrong held-out seed order')
  effective_pair=[]
  for label in ('hand','selected'):
   parameters=CONFIG['env_options'] if label=='hand' else selected_config['env_options']
   directory=OUTPUT/'heldout'/f'seed_{seed}'/label
   score,effective=check_run(directory,seed,parameters)
   expected_losses[label].append(score['loss_oe'])
   close(native[label+'_loss'],score['loss_oe'],'Held-out per-seed score',tol=1e-12)
   frames=[(torch.load(Path(score['run'])/'frame_states'/f'update_{u:07d}.pt',weights_only=True)[0,13]>=0).numpy() for u in [0,*UPDATES]]
   change=change_summary(frames)
   for k in ('changed_from_start','ongoing_change','late_change'):close(native[label+'_change'][k],change[k],'Held-out occupancy change',tol=1e-12)
   effective_pair.append(effective)
  a,b=effective_pair
  a=dict(a);b=dict(b);a.pop('env_options');b.pop('env_options')
  a.pop('output_dir',None);b.pop('output_dir',None)  # Destinations do not affect simulation.
  need(a==b,'Held-out conditions differ beyond food parameters')
 heldout=paired_heldout(expected_losses['hand'],expected_losses['selected'])
 native=load(OUTPUT/'heldout/summary.json')
 for field in ('hand_mean_loss','selected_mean_loss','paired_difference_mean','paired_difference_sample_std'):close(native[field],heldout[field],field,tol=1e-12)
 need(native['held_up']==heldout['held_up'] and len(native_rows)==3,'Wrong held-out decision/coverage')
else:
 need(best_id=='hand' and (OUTPUT/'heldout_skipped.json').is_file(),'Unexpectedly missing held-out evaluations')

# Independently verify all three displayed occupancy columns and scored-update labels against their source captures.
selected_directory=Path(selection['selected_artifacts'])
directories=[OUTPUT/'fixed',OUTPUT/'hand',selected_directory]
with Image.open(OUTPUT/'comparison.gif') as animation:
 need(animation.is_animated and animation.n_frames==11 and animation.size==(708,795),'Wrong GIF frames/grid')
 need(animation.info.get('loop')==0,'Wrong GIF loop')
 for i,u in enumerate([0,*UPDATES]):
  animation.seek(i);need(animation.info['duration']==400,'Wrong GIF timing')
  frame=np.array(animation.convert('RGB'))
  for col,directory in enumerate(directories):
   with Image.open(directory/'scoring_images'/f'update_{u:07d}.png') as im:expected=np.array(im.resize((212,212),Image.Resampling.NEAREST))[...,0]>127
   observed=frame[328:540,12+col*236:224+col*236].mean(-1)>127
   need(np.array_equal(expected,observed),'GIF occupancy mismatches source/update')
  if i in (0,5,10):animation.convert('RGB').save(OUTPUT.parent/'issue37_execution'/f'{OUTPUT.name}_gif_{u:04d}.png')
with Image.open(OUTPUT/'trajectory.png') as figure:figure.verify()
need(len(models)==len(fingerprints)==1,'Mixed model or simulator implementations')
runtime=load(OUTPUT.parent/'issue37_execution/protocol.json')['runtime']
need(next(iter(models))==runtime['model_weights_sha256'],'Unexpected model weights')
need(next(iter(fingerprints))==json.dumps(runtime['simulator_fingerprint'],sort_keys=True),'Unexpected simulator source')
expected_rollouts=4+3*BUDGET+(6 if completed['heldout_ran'] else 0)
need(len(rollouts)==expected_rollouts,'Unexpected rollout count')
report={'candidate_worlds_verified':BUDGET,'search_seed_rollouts_verified':3*BUDGET,'baseline_rollouts_verified':4,
 'heldout_rollouts_verified':6 if completed['heldout_ran'] else 0,'all_rollouts_verified':len(rollouts),
 'selected_id':best_id,'selection_objective':'minimize mean loss + 1.0 * sample std across seeds 303,7,13',
 'hand':hand,'selected':best,'heldout':heldout,'unique_effective_candidate_configs':len(groups),
 'duplicate_effective_configs':[{'parameters':dict(k),'candidate_ids':v} for k,v in groups.items() if len(v)>1],
 'gif':{'frames':11,'frame_duration_ms':400,'all_source_occupancy_panels_match':True},
 'model_weights_sha256':next(iter(models)),'single_corrected_simulator_fingerprint':True,
 'all_saved_embeddings_rescore_within_tolerance':True,'rollouts':rollouts}
path=OUTPUT.parent/'issue37_execution'/f'{OUTPUT.name}_validation.json'
path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('rollouts','duplicate_effective_configs')},indent=2))
print('VALIDATION',path)
