@ECHO OFF

REM attacks
git clone https://github.com/pralab/ImageNet-Patch

REM defense
git clone https://github.com/joellliu/SegmentAndComplete

REM recoverer
git clone https://github.com/facebookresearch/mae

REM cifar10 infer env
git clone https://github.com/Kahsolt/PyTorch_CIFAR10
PUSHD PyTorch_CIFAR10
IF NOT EXIST cifar10_models\state_dicts (
  python train.py --download_weights 1
)
POPD

ECHO Done!
ECHO.

PAUSE
