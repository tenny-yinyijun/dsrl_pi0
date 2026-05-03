  examples/scripts/collect_wm_testset.py (dsrl_pi0 venv)                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                
  Runs π₀/π₀.₅ in libero with stratified gaussian-noise scales — sigma round-robins through --noise-scales (default 0.5,1.0,2.0,3.0), so larger num-trajs give a denser, naturally success/failure-mixed eval set. Saves trajectories in the same libero_processed format the   
  live reward server already consumes.                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                
export DISPLAY=:0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
source /n/fs/iromdata/project/dsrl_pi0/.venv/bin/activate
# Mode 1 — seed (mild, original behavior; needs huge sigmas to break the policy)
python examples/scripts/collect_wm_testset.py \
--save-dir /n/fs/iromdata/project/shared/wm_testsets \
--name libero90_task57_seed --num-trajs 20 \
--noise-mode seed --noise-scales 1,2,3,4 \
--policy pi05 --task-suite libero_90 --task-id 57 --seed 0

# Mode 2 — action (cleanest disruption; sigma in env-action units)
python examples/scripts/collect_wm_testset.py \
--save-dir /n/fs/iromdata/project/shared/wm_testsets \
--name libero90_task57_action --num-trajs 20 \
--noise-mode action --noise-scales 0.05,0.15,0.3,0.6 \
--policy pi05 --task-suite libero_90 --task-id 57 --seed 0
                                                                                                                                                                                                                                                                            
# Mode 3 — obs (lies to the policy; sigma in mixed pos/rot/qpos units)
python examples/scripts/collect_wm_testset.py \
--save-dir /n/fs/iromdata/project/shared/wm_testsets \
--name libero90_task57_obs --num-trajs 20 \
--noise-mode obs --noise-scales 0.02,0.05,0.1,0.25 \
--policy pi05 --task-suite libero_90 --task-id 57 --seed 0
                                                                                                                                                                                                                                                                                
  Outputs:                                                                                                                                                                                                                                                                      
  - <save-dir>/<name>/annotation/train/<eid>.json + raw_videos/{agentview,wrist}/<eid>.mp4 (libero_processed)                                                                                                                                                                   
  - <save-dir>/<name>/manifest.json — per-eid noise_scale, is_success, env_steps, env_seed for stratified analysis later                                                                                                                                                        
  - Resumes cleanly: re-running with the same <save-dir>/<name> picks up after the highest existing eid.                                                                                                                                                                        
                                                                                                                                                                                                                                                                                
  examples/scripts/eval_wm_on_testset.py (open-world venv)                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                
  Loads one WM checkpoint, scores every traj using exactly the same code path as the live server (score_episode + _load_or_encode_latents + _load_actions), so eval and live can never silently diverge. Aggregates LPIPS overall, by success/failure, and by sigma; optionally 
  saves prediction mp4s.                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                
  cd /n/fs/iromdata/project/open-world                                                                                                                                                                                                                                        
  OPEN_WORLD_ROOT=$(pwd) CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python /n/fs/iromdata/project/dsrl_pi0/examples/scripts/eval_wm_on_testset.py \                                                                                                                                                                                     
      --testset-dir /n/fs/iromdata/project/shared/wm_testsets/libero90_task57_v1 \                                                                                                                                                                                              
      --ckpt-path /n/fs/iromdata/project/shared/playworld_rollout/<JOB_TAG>_full/wm_checkpoints/checkpoint-200.pt \                                                                                                                                                             
      --output /n/fs/iromdata/project/shared/wm_eval/ckpt-200_v1 \                                                                                                                                                                                                              
      --num-windows 4 --num-inference-steps 25 \                                                                                                                                                                                                                                
      --save-predictions                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                
  Outputs:                                                                                                                                                                                                                                                                    
  - <output>/per_traj.jsonl — one record per traj (eid, noise_scale, is_success, mean_lpips, per_frame_lpips, …)                                                                                                                                                                
  - <output>/aggregate.json — overall + by-success + by-sigma summary stats                                                                                                                                                                                                     
  - <output>/predictions/<eid>/{pred,gt}.mp4 if --save-predictions         
                                                                                                                                                                                                                                                                                
  The intended workflow: run the generator once to fix a test set, then re-run the eval against each WM checkpoint your loop produces (checkpoint-50.pt, -100.pt, -150.pt, …). Plotting aggregate.overall_lpips.mean vs WM step gives you the clean improvement curve the live  
  f-scores can't (since those entangle WM quality with policy drift). The by-sigma breakdown also tells you where improvement is happening — if mean_lpips drops uniformly across sigmas, the WM is genuinely better; if it only drops at low sigma, the WM is just learning    
  what the new policy does.  