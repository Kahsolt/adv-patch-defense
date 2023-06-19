@ECHO OFF
ECHO this file contains demo cmdline examples for your references, DO NOT run it directly
EXIT /B


:imagenet

REM clean
python run_adv_patch.py -M resnet50 -D imagenet

REM attack
python run_adv_patch.py -M resnet50 -D imagenet -B 32 --ap
python run_adv_patch.py -M resnet50 -D imagenet -B 32 --ap --ap_pgd
python run_adv_patch.py -M resnet50 -D imagenet -B 32 --ip
python run_adv_patch.py -M resnet50 -D imagenet -B 32 --ip --ip_idx 1

REM defense
python run_adv_patch.py -M resnet50 -D imagenet -B 4 --ap --sac
python run_adv_patch.py -M resnet50 -D imagenet -B 4 --ap --mae
python run_adv_patch.py -M resnet50 -D imagenet -B 4 --ap --sac --mae
python run_adv_patch.py -M resnet50 -D imagenet -B 4 --ap --sac --sac_complete --mae

REM run few batches
python run_adv_patch.py -M resnet50 -D imagenet -B 4 -L 8 --ap --sac --sac_complete --mae

REM visualize single image
python run_adv_patch.py -M resnet50 -D imagenet --idx 1 --ap --sac --sac_complete --mae --show --show_adv


:cifar10

REM clean
python run_adv_patch.py -M resnet50 -D cifar10

REM attack
python run_adv_patch.py -M resnet50 -D cifar10 --ap --sac --mae -B 4 --show
