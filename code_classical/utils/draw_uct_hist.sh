#!/bin/bash

python draw_ood_hist.py --dir_name cifar10-edl-test-draw --file_name max_prob_id_CIFAR10_ood_SVHN --ood_data SVHN --uncertainty MP
python draw_ood_hist.py --dir_name cifar10-edl-test-draw --file_name alpha0_id_CIFAR10_ood_SVHN --ood_data SVHN --uncertainty UM
python draw_ood_hist.py --dir_name cifar10-edl-test-draw --file_name diff_ent_id_CIFAR10_ood_SVHN --ood_data SVHN --uncertainty DE
python draw_ood_hist.py --dir_name cifar10-edl-test-draw --file_name mi_id_CIFAR10_ood_SVHN --ood_data SVHN --uncertainty MI
echo "EDL finished!"

python draw_ood_hist.py --dir_name cifar10-iedl-test-draw --file_name max_prob_id_CIFAR10_ood_SVHN --ood_data SVHN --uncertainty MP
python draw_ood_hist.py --dir_name cifar10-iedl-test-draw --file_name alpha0_id_CIFAR10_ood_SVHN --ood_data SVHN --uncertainty UM
python draw_ood_hist.py --dir_name cifar10-iedl-test-draw --file_name diff_ent_id_CIFAR10_ood_SVHN --ood_data SVHN --uncertainty DE
python draw_ood_hist.py --dir_name cifar10-iedl-test-draw --file_name mi_id_CIFAR10_ood_SVHN --ood_data SVHN --uncertainty MI
echo "I-EDL finished!"

python draw_ood_hist.py --dir_name cifar10-redl-test-draw --file_name max_prob_id_CIFAR10_ood_SVHN --ood_data SVHN --uncertainty MP
python draw_ood_hist.py --dir_name cifar10-redl-test-draw --file_name alpha0_id_CIFAR10_ood_SVHN --ood_data SVHN --uncertainty UM
python draw_ood_hist.py --dir_name cifar10-redl-test-draw --file_name diff_ent_id_CIFAR10_ood_SVHN --ood_data SVHN --uncertainty DE
python draw_ood_hist.py --dir_name cifar10-redl-test-draw --file_name mi_id_CIFAR10_ood_SVHN --ood_data SVHN --uncertainty MI
echo "R-EDL finished!"

python draw_ood_hist.py --dir_name cifar10-reedl-test-draw --file_name max_prob_id_CIFAR10_ood_SVHN --ood_data SVHN --uncertainty MP
python draw_ood_hist.py --dir_name cifar10-reedl-test-draw --file_name alpha0_id_CIFAR10_ood_SVHN --ood_data SVHN --uncertainty UM
python draw_ood_hist.py --dir_name cifar10-reedl-test-draw --file_name diff_ent_id_CIFAR10_ood_SVHN --ood_data SVHN --uncertainty DE
python draw_ood_hist.py --dir_name cifar10-reedl-test-draw --file_name mi_id_CIFAR10_ood_SVHN --ood_data SVHN --uncertainty MI
echo "Re-EDL finished!"
