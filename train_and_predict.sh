cd /home/lwidowski/PycharmProjects/NaiveBayes/
dataset=$1
/usr/bin/python3 configure.py $dataset
/usr/bin/python3 serve_data.py $dataset
echo "Next Step..."
/usr/bin/python3 Machine_Learning.py $dataset
echo "Next Step..."
/usr/bin/python3 Tableau_export.py $dataset
echo "Next Step..."
/usr/bin/python3 evaluation.py $dataset
