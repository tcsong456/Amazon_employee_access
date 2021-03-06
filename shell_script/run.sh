echo "creating addtional features"
python mysql_preparation.py
python main.py --task greedy_search_sql --num-of-tries 2 --instant-remove --cv-splits 5 --c 1.0
python main.py --task base_feature_sql --log

PIVOTTABLE_OPTIONS="all train test"
BODY_OPTIONS=("extra" "tree" "meta" "base")
TT_OPTIONS=("tuple" "tripple" "basic")

for po in $PIVOTTABLE_OPTIONS;do
  python main.py --task build_pivottable --use-numpy --use-split $po
done

for ((i=0;i<${#BODY_OPTIONS[@]};++i));do
  split=${BODY_OPTIONS[i]}
  python main.py --task build_body --use-numpy --option $split
done

for tt in ${TT_OPTIONS[@]};do
  python main.py --task build_tuple_tripple --use-numpy --option $tt
done

python main.py --task build_dataset --use-numpy

python main.py --mode produce_result --pattern $1 --task stacking --num-iter 1 --stacking-splits 4  --use-numpy
