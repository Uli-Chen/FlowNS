# Activate environment                                                                                     
  conda activate pytorch
                                                                                                             
  # --- Baselines ---                                                                                        
  python scripts/run_baseline.py --config configs/baselines/rns_mf_mindsmall.yaml --model BPR
  python scripts/run_baseline.py --config configs/baselines/dns_mf_mindsmall.yaml --model BPR                
  python scripts/run_baseline.py --config configs/baselines/pns_mf_mindsmall.yaml --model BPR
                                                                                                             
  # --- FlowNeg ---
  python scripts/run_flowneg.py --config configs/flowneg_mf_mindsmall.yaml --model BPR                       
  python scripts/run_flowneg.py --config configs/flowneg_lgcn_mindsmall.yaml --model LightGCN                
   
  # --- Options ---                                                                                          
  --seed 42              # reproducibility (default 42)
  --output_dir results   # where to save JSON results (default)                                              
                                                                                                             
  Results are saved as JSON in results/.