# Training Runs Comparison

Comprehensive comparison of all PPO training runs for navigation in Gazebo environments.

**Last Updated:** January 24, 2026 15:20 UTC

---

## Runs Table

| Run Name | Date | World | Vision | Sampler | Reach | Dist | Reward | Timesteps | Success % | Collision % | Mean Reward | Checkpoints | Notes |
|----------|------|-------|--------|---------|-------|------|--------|-----------|-----------|-------------|-------------|-------------|-------|
| 20260123_2154_small_house_resnet18_mapSampler_reachBFS_distUniform_rewardV1_T200k | 2026-01-23 21:54 | small_house | resnet18 | mapSampler | reachBFS | distUniform | V1 | 201,721 | 39.3% (peak@iter20) | 46.4% (best@iter20) | 831.3 (peak@iter24) | 13 saved (every 2 iters) | **Steps=1 bug fix applied**. Peak at iter 20-24, then degraded. Recommend using iter20/24 checkpoint. |
| 20260124_1513_small_house_dinov2vits14_mapSampler_reachBFS_distUniform_rewardV1_T200k | 2026-01-24 15:13 | small_house | dinov2_vits14 | mapSampler | reachBFS | distUniform | V1 | 200,000 (target) | TBD | TBD | TBD | TBD | **IN PROGRESS** - DINOv2 ViT-S/14 (384-dim), same settings as ResNet18. Estimated completion: ~40 hours. |

---

See `TRAINING_RESULTS_small_house_resnet18_200k_steps1fix.md` for detailed ResNet18 analysis.
