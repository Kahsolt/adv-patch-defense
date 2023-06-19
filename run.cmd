@ECHO OFF
ECHO this file contains demo cmdline examples for your references, DO NOT run it directly
EXIT /B


:imagenet

REM clean
python run_adv_patch.py -M resnet50 -D imagenet

REM attack
python run_adv_patch.py -M resnet50 -D imagenet --ap
python run_adv_patch.py -M resnet50 -D imagenet --ap --ap_pgd
python run_adv_patch.py -M resnet50 -D imagenet --ip
python run_adv_patch.py -M resnet50 -D imagenet --ip --ip_idx 1

REM defense
python run_adv_patch.py -M resnet50 -D imagenet --ap --sac
python run_adv_patch.py -M resnet50 -D imagenet --ap --mae
python run_adv_patch.py -M resnet50 -D imagenet --ap --sac --mae
python run_adv_patch.py -M resnet50 -D imagenet --ap --sac --sac_complete --mae

python run_adv_patch.py -M resnet50 -D imagenet --ip --ip_idx 1 --sac --sac_complete --mae

REM run few batches
python run_adv_patch.py -M resnet50 -D imagenet -L 8 --ap --sac --sac_complete --mae

REM visualize single image
python run_adv_patch.py -M resnet50 -D imagenet --idx 1 --ap --sac --sac_complete --mae --show --show_adv


:cifar10

REM clean
python run_adv_patch.py -M resnet50 -D cifar10

REM attack
python run_adv_patch.py -M resnet50 -D cifar10 --ap

REM defense
python run_adv_patch.py -M resnet50 -D cifar10 --ap --mae

REM visualize single image
python run_adv_patch.py -M resnet50 -D cifar10 --idx 1 --ap --mae --show --show_adv
