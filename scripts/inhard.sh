python main_use.py --method "st_adapter" --st_adapt --lr 0.001 --comm_roun 100 --DATASET "INHARD" --nb_classes 14 --data_path "./inhard_anno/" --client_number 16 --client_sample 0.25 --partition_alpha 1.0
python main_use.py --method "scratch" --fulltune --lr 0.001 --comm_roun 100 --DATASET "INHARD" --nb_classes 14 --data_path "./inhard_anno/" --client_number 16 --client_sample 0.25 --partition_alpha 1.0
python main_use.py --method "scratch" --fulltune --lr 0.001 --comm_roun 100 --DATASET "INHARD" --nb_classes 14 --data_path "./inhard_anno/" --client_number 16 --partition_alpha 1.5
python main_use.py --method "linear_head" --lr 0.0005 --comm_roun 100 --DATASET "INHARD" --nb_classes 14 --data_path "./inhard_anno/" --client_number 16 --client_sample 0.25 partition_alpha 1.5
python main_use.py --method "bias" --lr 0.0005 --comm_roun 100 --DATASET "INHARD" --nb_classes 14 --data_path "./inhard_anno/" --client_number 16 --client_sample 0.25 partition_alpha 1.5
python main_use.py --method "linear_head" --lr 0.0005 --comm_roun 100 --DATASET "INHARD" --nb_classes 14 --data_path "./inhard_anno/" --client_number 16 --client_sample 0.25 --partition_alpha 1.0
python main_use.py --method "st_adapter" --st_adapt --lr 0.001 --comm_roun 100 --DATASET "INHARD" --nb_classes 14 --data_path "./inhard_anno/" --client_number 16 --client_sample 0.25 --partition_alpha 1.0
python main_use.py --method "bias" --lr 0.001 --comm_roun 100 --DATASET "INHARD" --nb_classes 14 --data_path "./inhard_anno/" --client_number 16 --client_sample 0.25 --partition_alpha 1.0
python main_use.py --method "scratch" --fulltune--lr 0.001 --comm_roun 100 --DATASET "INHARD" --nb_classes 14 --data_path "./inhard_anno/" --client_number 16 --client_sample 0.25 partition_alpha 1.5
python main_use.py --method "linear_head" --lr 0.0005 --comm_roun 100 --DATASET "INHARD" --nb_classes 14 --data_path "./inhard_anno/" --client_number 16 partition_alpha 1.5
python main_use.py --method "st_adapter" --st_adapt --lr 0.001 --comm_roun 100 --DATASET "INHARD" --nb_classes 14 --data_path "./inhard_anno/" --client_number 16 partition_alpha 1.5
python main_use.py --method "bias" --lr 0.0005 --comm_roun 100 --DATASET "INHARD" --nb_classes 14 --data_path "./inhard_anno/" --client_number 16 partition_alpha 1.5
python main_use.py --method "linear_head" --lr 0.0005 --comm_roun 100 --DATASET "INHARD" --nb_classes 14 --data_path "./inhard_anno/" --client_number 16 --partition_alpha 1.0
python main_use.py --method "st_adapter" --st_adapt --lr 0.001 --comm_roun 100 --DATASET "INHARD" --nb_classes 14 --data_path "./inhard_anno/" --client_number 16 --partition_alpha 1.0
python main_use.py --method "bias" --lr 0.0005 --comm_roun 100 --DATASET "INHARD" --nb_classes 14 --data_path "./inhard_anno/" --client_number 16 --partition_alpha 1.0
python main_use.py --method "scratch" --fulltune --lr 0.001 --comm_roun 100 --DATASET "INHARD" --nb_classes 14 --data_path "./inhard_anno/" --client_number 16 --partition_alpha 1.0
