# Official implementation of LiDER: *On the Effectiveness of Lipschitz-Driven Rehearsal in Continual Learning*

*Accepted at NeurIPS 2022*

Based on [https://github.com/aimagelab/mammoth](Mammoth)

## Citation

**Please stand by.**

## Example:

Command:

`python utils/main.py --dataset=seq-cifar100 --model=er_ace_lipschitz --n_epochs=50 --buffer_size=500 --lr=0.1 --load_cp=checkpoints/erace_pret_on_tinyr.pth --pre_epochs=200 --datasetS=tinyimgR --non_verbose`
