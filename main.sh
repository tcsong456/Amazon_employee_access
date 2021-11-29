echo "Running project amazon employee access prediction"
bash shell_script/preprocess.sh

if [ $# -eq 0 ]
then
  echo 'must specify the pattern when producing result'
  exit 1
fi

bash shell_script/run.sh $1
