cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir/../..

cd $root_dir

redo=1
data_root_dir="$HOME/data/VOCdevkit"
dataset_name="VOC0712Plus"
mapfile="$root_dir/data/$dataset_name/labelmap_voc.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi

python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/data/$dataset_name/trainval.txt $data_root_dir/$dataset_name/$db/$dataset_name"_trainval_"$db examples/$dataset_name
python $root_dir/scripts/create_annoset.py --anno-type=None --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/data/$dataset_name/test.txt $data_root_dir/$dataset_name/$db/$dataset_name"_test_"$db examples/$dataset_name

