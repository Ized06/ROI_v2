CAUSE=crossing_pedestrian
EXPERIMENT_NAME=2021-2-6_054709_w_dataAug_attn_baseline_reimplemented_implementation
EPOCH=7


python visualization.py \
--model ./snapshots/$CAUSE/$EXPERIMENT_NAME/inputs-camera-epoch-$EPOCH.pth \
--cause $CAUSE
