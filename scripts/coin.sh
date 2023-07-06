python main_use.py --method "bias" --lr 0.0005 --comm_round 20 --DATASET "COIN" --nb_classes 180 --data_path "./coin_anno" --client_number 16 --partition_alpha 0.5
python main_use.py --method "scratch" --fulltune --lr 0.001 --comm_round 20 --DATASET "COIN" --nb_classes 180 --data_path "./coin_anno" --client_number 16 --partition_alpha 0.5
python main_use.py --method "st_adapter" --st_adapt --lr 0.001 --comm_round 20 --DATASET "COIN" --nb_classes 180 --data_path "./coin_anno" --client_number 16 --client_sample 0.25 --partition_alpha 0.1
python main_use.py --method "scratch" --fulltune --lr 0.001 --comm_round 20 --DATASET "COIN" --nb_classes 180 --data_path "./coin_anno" --client_number 16 --client_sample 0.25 --partition_alpha 0.1
python main_use.py --method "linear_head" --lr 0.0005 --comm_round 20 --DATASET "COIN" --nb_classes 180 --data_path "./coin_anno" --client_number 16 --client_sample 0.25 --partition_alpha 0.1
python main_use.py --method "bias" --lr 0.001 --comm_round 20 --DATASET "COIN" --nb_classes 180 --data_path "./coin_anno" --client_number 16 --client_sample 0.25 --partition_alpha 0.1
python main_use.py --method "linear_head" --lr 0.0005 --comm_round 20 --DATASET "COIN" --nb_classes 180 --data_path "./coin_anno" --client_number 16 --partition_alpha 0.1
python main_use.py --method "st_adapter" --st_adapt --lr 0.001 --comm_round 20 --DATASET "COIN" --nb_classes 180 --data_path "./coin_anno" --client_number 16 --partition_alpha 0.1
python main_use.py --method "bias" --lr 0.0005 --comm_round 20 --DATASET "COIN" --nb_classes 180 --data_path "./coin_anno" --client_number 16 --partition_alpha 0.1
python main_use.py --method "scratch" --fulltune --lr 0.001 --comm_round 20 --DATASET "COIN" --nb_classes 180 --data_path "./coin_anno" --client_number 16 --partition_alpha 0.1

