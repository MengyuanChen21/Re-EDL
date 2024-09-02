cd ..

CUDA_VISIBLE_DEVICES=0 python save_uct_for_drawing.py --configid "2_cifar10/cifar10-edl-test-draw" --suffix test
CUDA_VISIBLE_DEVICES=0 python save_uct_for_drawing.py --configid "2_cifar10/cifar10-iedl-test-draw" --suffix test
CUDA_VISIBLE_DEVICES=0 python save_uct_for_drawing.py --configid "2_cifar10/cifar10-redl-test-draw" --suffix test
CUDA_VISIBLE_DEVICES=0 python save_uct_for_drawing.py --configid "2_cifar10/cifar10-reedl-test-draw" --suffix test
