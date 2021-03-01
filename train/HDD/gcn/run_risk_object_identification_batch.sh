for i in {2..2}
do
    EXPERIMENT_NAME=2021-2-8_134415_w_dataAug_attn_augmented_data_baseline_both_vehicle_and_human
    EPOCH=$i


    CUDA_VISIBLE_DEVICES=0,1 python eval_intervention_test.py --cause crossing_vehicle \
    --model ./snapshots/crossing_pedestrian/$EXPERIMENT_NAME/inputs-camera-epoch-$EPOCH.pth \
    --fusion attn

done