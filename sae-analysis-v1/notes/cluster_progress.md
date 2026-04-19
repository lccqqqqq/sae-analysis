# Full 50-Batch Run — Progress Log

Started: 2026-04-15 ~07:25 BST. 5 jobs submitted across sondhigpu (Gemma) and gpulong (rest).

## Jobs

| Job | Preset | Queue | GPU | Layers | ETA |
|---|---|---|---|---|---|
| 6649441 | gemma-2-2b | sondhigpu-priority | H200 141G | 26 | ~6-8 h |
| 6649442 | gpt2-small | gpulong | rtx4090 24G | 12 | ~80 min |
| 6649443 | llama-3.2-1b | gpulong | rtx4090 24G | 9 | ~2 h |
| 6649444 | qwen2-0.5b | gpulong | rtx3090 24G | 12 | ~4-5 h |
| 6649445 | pythia-70m | gpulong | rtx2080 12G | 6 | ~20 min |

## Updates

## 2026-04-15 07:29 BST
- **6649441 gemma-2-2b** [PENDING 00:00:00]: 
- **6649442 gpt2-small** [RUNNING 00:01:57]: [INFO] Loading sae_lens SAE: release=gpt2-small-resid-post-v5-32k hook=blocks.7.hook_resid_post
- **6649443 llama-3.2-1b** [RUNNING 00:01:57]: [INFO] Loading sae_lens SAE: release=chanind/sae-llama-3.2-1b-res hook=blocks.6.hook_resid_post
- **6649444 qwen2-0.5b** [PENDING 00:00:00]: 
- **6649445 pythia-70m** [RUNNING 00:01:57]: [batch    3/50 start= 169792] dt=16.09s avg=20.11s eta=15m45s gpu_mem=1.1G peak=1.3G | L0:n=8,fH=1.42,tH=1.20 L1:n=15,fH=2.44,tH=1.87 L2:n=54,fH=2.47,tH=2.25 L3:n=42,fH=2.44,tH=2.13 L4:n=38,fH=1.87,tH=1.91 L5:n=277,fH=1.93,tH=1.98


## 2026-04-15 19:19 BST
- **6649441 gemma-2-2b** [RUNNING 07:09:01]: [batch   37/50 start=1856768] dt=1127.27s avg=674.73s eta=2h26m11s gpu_mem=24.7G peak=28.3G | L0:n=11,fH=0.55,tH=0.16 L1:n=21,fH=1.21,tH=0.66 L2:n=29,fH=1.65,tH=1.10 L3:n=22,fH=2.27,tH=1.61 L4:n=41,fH=2.87,tH=2.22 L5:n=27,fH=3.34,tH=2.94 L6:n=38,fH=3.72,tH=3.46 L7:n=33,fH=4.36,tH=4.18 L8:n=29,fH=4.46,tH=4.51 L9:n=66,fH=4.81,tH=4.88 L10:n=80,fH=4.94,tH=5.04 L11:n=69,fH=5.10,tH=5.19 L12:n=75,fH=5.10,tH=5.22 L13:n=65,fH=5.20,tH=5.40 L14:n=81,fH=5.27,tH=5.43 L15:n=59,fH=5.22,tH=5.42 L16:n=71,fH=5.23,tH=5.42 L17:n=48,fH=5.24,tH=5.41 L18:n=39,fH=5.22,tH=5.40 L19:n=30,fH=5.23,tH=5.36 L20:n=32,fH=5.25,tH=5.37 L21:n=37,fH=5.28,tH=5.38 L22:n=48,fH=5.21,tH=5.28 L23:n=31,fH=5.23,tH=5.22 L24:n=29,fH=5.17,tH=5.20 L25:n=71,fH=5.12,tH=5.13
- **6649442 gpt2-small** [COMPLETED 00:58:53]: [INFO] L11: saved /mnt/users/clin/workspace/sae-analysis/sae-analysis-v1/data/gpt2-small/20260415_072737/entropy_comparison_resid_post_layer11.pt
- **6649443 llama-3.2-1b** [COMPLETED 02:04:53]: [INFO] L8: saved /mnt/users/clin/workspace/sae-analysis/sae-analysis-v1/data/llama-3.2-1b/20260415_072743/entropy_comparison_resid_layer8.pt
- **6649444 qwen2-0.5b** [FAILED 00:00:00]: 
- **6649445 pythia-70m** [COMPLETED 00:20:28]: [INFO] L5: saved /mnt/users/clin/workspace/sae-analysis/sae-analysis-v1/data/pythia-70m/20260415_072734/entropy_comparison_resid_out_layer5.pt


## 2026-04-15 19:22 BST (morning)
- **6649441 gemma-2-2b** [   RUNNING   07:11:46 ]: [batch   37/50 start=1856768] dt=1127.27s avg=674.73s eta=2h26m11s gpu_mem=24.7G peak=28.3G | L0:n=11,fH=0.55,tH=0.16 L1:n=21,fH=1.21,tH=0.66 L2:n=29,fH=1.65,tH=1.10 L3:n=22,fH=2.27,tH=1.61 L4:n=41,fH=2.87,tH=2.22 L5:n=27,fH=3.34,tH=2.94 L6
- **6649442 gpt2-small** [ COMPLETED   00:58:53 ]: [INFO] L11: saved /mnt/users/clin/workspace/sae-analysis/sae-analysis-v1/data/gpt2-small/20260415_072737/entropy_comparison_resid_post_layer11.pt
- **6649443 llama-3.2-1b** [ COMPLETED   02:04:53 ]: [INFO] L8: saved /mnt/users/clin/workspace/sae-analysis/sae-analysis-v1/data/llama-3.2-1b/20260415_072743/entropy_comparison_resid_layer8.pt
- **6820129 qwen2-0.5b** [   RUNNING   00:01:42 ]: 
- **6649445 pythia-70m** [ COMPLETED   00:20:28 ]: [INFO] L5: saved /mnt/users/clin/workspace/sae-analysis/sae-analysis-v1/data/pythia-70m/20260415_072734/entropy_comparison_resid_out_layer5.pt
