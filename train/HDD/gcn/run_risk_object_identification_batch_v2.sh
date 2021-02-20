for i in {1..20}
do
    EXPERIMENT_NAME=2021-2-16_083214_w_dataAug_attn
    EPOCH=$i


    CUDA_VISIBLE_DEVICES=0,1 python eval_intervention_test.py --cause crossing_vehicle \
    --model ./snapshots/crossing_vehicle/$EXPERIMENT_NAME/inputs-camera-epoch-$EPOCH.pth \
    --fusion attn

done