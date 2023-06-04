# adv-patch-defense

    Adversarial Patch defense using SegmentAndComplete (SAC) & Masked AutoEncoder (MAE)

----

Naive defense pipeline to AdvPatch := SAC + MAE

  - Consider image classification and object dectection task
  - AdvPatch puts high-freq color patches on given image (circle/square, ~15%(?) full area)
  - SAC detects adv patches then mask with pure black (round-corner square, nearly cover)
  - MAE recovers the black area

Partial result (`limit=256`):

| Setting | Accuracy | cmdline |
| :-: | :-: | :-: | :-: |
| clean   | 96.484% | `python run_adv_patch.py -L 256 -B 32` |
| attack  | 44.531% | `python run_adv_patch.py -L 256 -B 32 --ap` |
| attack  | 46.484% | `python run_adv_patch.py -L 256 -B 32 --ap --ap_pgd` |
| defense |         | `python run_adv_patch.py -L 256 -B 4 --ap --sac --mae` |
| defense |         | `python run_adv_patch.py -L 256 -B 4 --ap --sac --sac_complete --mae` |

⚠ For 12G VRAM, max `batch_size` for `attack/defense` is `48/6` separately
⚠ To defend against `--ip` attack, you must turn on `--sac_complete` because the pre-generated patches are not that PGD-like noisy

⚪ run error trouble shoot

Q: pip package version issues  
A: assure `timm==0.3.2`, `torch>1.7`; recommend `numpy==1.24.3`, `torch==2.0.1+cu117`, `torchvision==0.15.2+cu117`  

Q: ImportError in site-package `timm`  
A: modify `timm/models/layers/helper.py:6` to `import collections.abc as container_abcs`  

Q: deprecation numpy error in local `mae` repo  
A: modify `mae/util/pos_embed.py:56`, change `np.float` to `np.float32`  


### quick start

- run `repo/init_repos_mini.cmd`
- download [MAE weights](https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth) to `repo/mae/models`
- download test data [ImageNet-1k]()
- run clean test: `python run_adv_patch.py`
- run attack test: `python run_adv_patch.py --ap` or `python run_adv_patch.py --ip`
- run defense test
  - batch: `python run_adv_patch.py -B 16 -L 16 --ap --sac --mae`
  - all:   `python run_adv_patch.py --ap --sac --mae`


pipeline subsitutes:

```
⚪ adv patch attack
  - AdvPatch (*)
  - ImageNet-Patch
  - DPatch / Robust-DPatch
  - regioned PGD
  - feature_level_adv

⚪ adv patch detect (mask / drop)
  - SAC (*)
  - smoothed-vit
  - PatchCleanser

⚪ high-freq suppress
  - PatchGuard
  - local gradients smoothing
  - unsharp mask (new = low + high * alpha)

⚪ image recover
  - MAE / dMAE (*)
  - MST / ZITS inpainting
  - DDNM (null-space diffusion inpainting)
```

### ImageNet-Patch info

```python
{
  'patch_size': 50, 
  'target_names': {
    804: 'soap dispenser', 
    513: 'cornet', 
    923: 'plate', 
    954: 'banana', 
    968: 'cup', 
    878: 'typewriter keyboard', 
    546: 'electric guitar', 
    585: 'hair spray', 
    806: 'sock', 
    487: 'cellular telephone'}, 
  'patch_type': 'square', 
  'input_shape': (3, 224, 224)
}
```

### reference

⚪ adv attack toolbox

- adversarial-robustness-toolbox: [https://github.com/Trusted-AI/adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
  - 很全面的攻防框架！！
- cleverhans: [https://github.com/cleverhans-lab/cleverhans](https://github.com/cleverhans-lab/cleverhans)
  - 已停止更新，实现的攻击比 torchattacks 还少

⚪ patch attack

- ImageNet-Patch: [https://github.com/pralab/ImageNet-Patch](https://github.com/pralab/ImageNet-Patch)
  - 基于 ImageNet 预制作的 10 类 patch
- AdvPattern: [https://github.com/whuAdv/AdvPattern](https://github.com/whuAdv/AdvPattern)
  - 攻击行人重识别模型，衣服上贴补丁，不知所云的方法
- feature_level_adv: [https://github.com/thestephencasper/feature_level_adv](https://github.com/thestephencasper/feature_level_adv)
  - 使用GAN去生成带补丁的对抗样本，欺骗一群分类器；很鸡贼的数据来源

⚪ image recover (patch & inpaint)

- mae: [https://github.com/facebookresearch/mae](https://github.com/facebookresearch/mae)
- dmae: [https://github.com/quanlin-wu/dmae](https://github.com/quanlin-wu/dmae)
  - 数据集上加高斯降噪再训练的MAE, 没用GANLoss，恢复效果很烂
- MST_inpainting: [https://github.com/ewrfcas/MST_inpainting](https://github.com/ewrfcas/MST_inpainting)
- ZITS_inpainting: [https://github.com/DQiaole/ZITS_inpainting](https://github.com/DQiaole/ZITS_inpainting)
  - Transformer 擦除补全模型

⚪ defense

- phattacks: [https://github.com/tongwu2020/phattacks](https://github.com/tongwu2020/phattacks)
  - 物理防御，使用区块化PGD的对抗训练；水文
- smoothed-vit: [https://github.com/MadryLab/smoothed-vit](https://github.com/MadryLab/smoothed-vit)
  - 对抗补丁防御：原图纵向切竖条，对每个竖条过VIT做分类，最后投票聚合
- SegmentAndComplete: [https://github.com/joellliu/SegmentAndComplete](https://github.com/joellliu/SegmentAndComplete)
  - 对抗补丁位置检测，移除
- PatchGuard： [https://github.com/inspire-group/PatchGuard](https://github.com/inspire-group/PatchGuard)
  - 使用更小的感受野，特征图数值下截断0、大激活值直接改成0
- PatchCleanser: [https://github.com/inspire-group/PatchCleanser](https://github.com/inspire-group/PatchCleanser)
  - 用一个比攻击者补丁更大的补丁轮询遮盖原图，若所有输出不完全一致则保持第一个补丁再用第二个补丁轮询；纯理论意淫，sb工作
- local_gradients_smoothing: [https://github.com/fabiobrau/local_gradients_smoothing](https://github.com/fabiobrau/local_gradients_smoothing)
  - 传统数值方法(Sobel)计算一张图的局域梯度，修改原图压抑梯度大的部分（线性，梯度越大被压抑到越小）
  - 开源代码似乎并未实现论文全部演示

----

by Armit
2023/03/23 
