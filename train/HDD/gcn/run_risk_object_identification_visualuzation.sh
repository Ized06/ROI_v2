CAUSE=crossing_vehicle
EXPERIMENT_NAME=2021-2-16_144547_w_dataAug_attn
EPOCH=1


python visualization.py \
--model ./snapshots/$CAUSE/$EXPERIMENT_NAME/inputs-camera-epoch-$EPOCH.pth \
--cause $CAUSE
