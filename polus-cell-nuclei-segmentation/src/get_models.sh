curl -L -o dsb2018_topcoders.zip https://www.dropbox.com/s/qvtgbz0bnskn9wu/dsb2018_topcoders.zip

unzip dsb2018_topcoders.zip albu/weights/* -d ./dsb2018_topcoders/
unzip dsb2018_topcoders.zip selim/nn_models/* -d ./dsb2018_topcoders/
unzip dsb2018_topcoders.zip victor/nn_models/* -d ./dsb2018_topcoders/
unzip dsb2018_topcoders.zip victor/lgbm_models/* -d ./dsb2018_topcoders/
unzip dsb2018_topcoders.zip albu/data/folds.csv -d ./dsb2018_topcoders/

rm dsb2018_topcoders.zip