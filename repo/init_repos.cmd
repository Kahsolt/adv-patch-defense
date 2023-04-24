@ECHO OFF

REM attacks
git clone https://github.com/tongwu2020/phattacks
git clone https://github.com/Trusted-AI/adversarial-robustness-toolbox
git clone https://github.com/whuAdv/AdvPattern
git clone https://github.com/thestephencasper/feature_level_adv
git clone https://github.com/cleverhans-lab/cleverhans
git clone https://github.com/fabiobrau/local_gradients_smoothing

REM defense
git clone https://github.com/MadryLab/smoothed-vit
git clone https://github.com/joellliu/SegmentAndComplete
git clone https://github.com/inspire-group/PatchGuard
git clone https://github.com/inspire-group/PatchCleanser

REM recoverer
git clone https://github.com/facebookresearch/mae
git clone https://github.com/quanlin-wu/dmae
git clone https://github.com/ewrfcas/MST_inpainting
git clone https://github.com/DQiaole/ZITS_inpainting
git clone https://github.com/wyhuai/DDNM

ECHO Done!
ECHO.

PAUSE
