CAUSE=crossing_vehicle
EXPERIMENT_NAME=2021-2-16_083214_w_dataAug_attn_re_implemented_baseline
EPOCH=7


python visualization.py \
--model ./snapshots/$CAUSE/$EXPERIMENT_NAME/inputs-camera-epoch-$EPOCH.pth \
--cause $CAUSE
