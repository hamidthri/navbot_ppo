# Vision-based SAC Navigation - FIXED VERSION

## Key Fixes Applied

### 1. **ImageNet Normalization (CRITICAL)**
- **Problem**: ResNet was trained on ImageNet with normalized inputs (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Fix**: Added normalization in `vision_backbones_improved.py`
- **Impact**: This alone can cause 50-100% performance difference

### 2. **TensorBoard Logging**
- Added comprehensive logging to monitor training progress
- Logs saved in `models/sac_vision/{fusion}_{backbone}/logs/`
- Tracks: rewards, success rate, losses, Q-values, buffer size

### 3. **Configurable Fusion**
- Made fusion type a command-line argument
- Test different fusion strategies: concat, film, attention, gated

### 4. **Better Hyperparameters**
- Increased batch size to 128 (from 64) for more stable vision learning
- Increased buffer size to 100k
- Better default timesteps (100k instead of 10k for meaningful training)

## File Mapping

### Use These Improved Files:
- `sac_training_vision_improved.py` → Main training script
- `vision_backbones_improved.py` → Vision backbones with normalization
- `sac_networks_vision_improved.py` → Networks with configurable fusion
- `sac_vision_improved.py` → SAC agent with fusion parameter

### Keep These Files (they're fine):
- `environment_small_house.py` → Environment
- `replay_buffer_vision.py` → Replay buffer
- `gate_fusion.py` → Fusion modules
- `small_house_region_sampler.py` → Position sampler

## Quick Start

### 1. Basic Training (Concat Fusion, ResNet-18)
```bash
python3 sac_training_vision_improved.py \
    --fusion_type concat \
    --backbone resnet18 \
    --max_timesteps 100000
```

### 2. Try Different Fusion (FiLM)
```bash
python3 sac_training_vision_improved.py \
    --fusion_type film \
    --backbone resnet18 \
    --max_timesteps 100000
```

### 3. Monitor with TensorBoard
```bash
tensorboard --logdir models/sac_vision/
```

### 4. Full Training Run (500k timesteps)
```bash
python3 sac_training_vision_improved.py \
    --fusion_type concat \
    --backbone resnet18 \
    --max_timesteps 500000 \
    --batch_size 128 \
    --buffer_size 100000
```

## Expected Results

With the fixes:
- **10k timesteps**: Should see some learning (not converged yet)
- **50k timesteps**: Success rate should improve noticeably
- **100k+ timesteps**: Should achieve reasonable navigation

Without ImageNet normalization, the vision features are essentially random noise!

## All Arguments

```
--backbone          Vision backbone (resnet18, resnet34, resnet50, simple_cnn)
--fusion_type       Fusion module (concat, film, attention, gated)
--hidden_dim        Hidden layer size (default: 256)
--max_timesteps     Total training steps (default: 100000)
--start_timesteps   Random exploration steps (default: 2000)
--update_after      Start updates after N steps (default: 2000)
--update_every      Update frequency (default: 50)
--batch_size        Batch size (default: 128)
--buffer_size       Replay buffer capacity (default: 100000)
--lr_actor          Actor learning rate (default: 3e-4)
--lr_critic         Critic learning rate (default: 3e-4)
--lr_alpha          Alpha learning rate (default: 3e-4)
--gamma             Discount factor (default: 0.99)
--tau               Target network update rate (default: 0.005)
--save_dir          Save directory (default: models/sac_vision)
--save_freq         Save every N steps (default: 10000)
```

## Troubleshooting

### Still not learning?
1. Check TensorBoard - is Q-value increasing?
2. Verify buffer is filling (check buffer_size in logs)
3. Try simpler fusion first (concat before film/attention)
4. Increase max_timesteps (vision needs more data)

### Out of memory?
1. Reduce batch_size (try 64)
2. Reduce buffer_size (try 50000)
3. Use resnet18 instead of resnet50

### Training too slow?
1. Use GPU (automatic if available)
2. Reduce update_every (but may be less stable)
3. Use simpler backbone (simple_cnn)

## Experiment Recommendations

1. **Baseline**: Start with concat fusion, resnet18, 100k timesteps
2. **Compare fusions**: Test film, attention, gated with same settings
3. **Longer training**: Run best fusion for 500k timesteps
4. **Architecture search**: Try resnet50 if resnet18 works well