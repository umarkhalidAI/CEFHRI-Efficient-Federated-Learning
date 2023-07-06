python main_use.py --method "st_adapter" --st_adapt --lr 0.001 --comm_round 100 --DATASET "HRI" --nb_classes 30 --data_path "./HRI_anno/" --client_number 16 --client_sample 0.25
python main_use.py --method "scratch" --fulltune --lr 0.001 --comm_round 100 --DATASET "HRI" --nb_classes 30 --data_path "./HRI_anno/" --client_number 16
python main_use.py --method "linear_head" --lr 0.0005 --comm_round 100 --DATASET "HRI" --nb_classes 30 --data_path "./HRI_anno/" --client_number 16 --client_sample 0.25
python main_use.py --method "bias" --lr 0.0005 --comm_round 100 --DATASET "HRI" --nb_classes 30 --data_path "./HRI_anno/" --client_number 16 --client_sample 0.25
python main_use.py --method "scratch" --fulltune --lr 0.001 --comm_round 100 --DATASET "HRI" --nb_classes 30 --data_path "./HRI_anno/" --client_number 16 --client_sample 0.25
python main_use.py --method "linear_head" --lr 0.0005 --comm_round 100 --DATASET "HRI" --nb_classes 30 --data_path "./HRI_anno/" --client_number 16 --client_sample 0.25 --partition_alpha 0.5
python main_use.py --method "st_adapter" --st_adapt --lr 0.001 --comm_round 100 --DATASET "HRI" --nb_classes 30 --data_path "./HRI_anno/" --client_number 16 --client_sample 0.25 --partition_alpha 0.5
python main_use.py --method "bias" --lr 0.001 --comm_round 100 --DATASET "HRI" --nb_classes 30 --data_path "./HRI_anno/" --client_number 16 --client_sample 0.25 --partition_alpha 0.5
