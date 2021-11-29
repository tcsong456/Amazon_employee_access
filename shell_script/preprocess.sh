if [ ! -f amazon-employee-access-challenge.zip ]
then
  echo "downloading data from kaggle"
  kaggle competitions download -c amazon-employee-access-challenge
else
  echo "data has already been downloaded"
fi

if [ ! -d data ]
then
   mkdir data
fi

unzip -o amazon-employee-access-challenge.zip -d data

