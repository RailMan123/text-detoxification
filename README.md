## Rail Sayfeev
## r.sayfeev@innopolis.university
## B21-DS-01

# Usage
1) Firstly, clone the directory
and install requirements.
2) Then, move to src/data and run make_dataset.py
to download and generate the data.
3) Then you can start training by going to
src/models and running train_model.py
(ONLY if you have GPU on your device)
4) After training, you can detox sentences by running predict_model.py.
Wait until program asks you to input your sentence. Then type it and get your neutralized
version of sentence.
5) However, if you do not want to perform all these steps to see results,
you can just go to notebooks/ directory and go through 2.0_final_training.ipynb notebook to 
see training and prediction of the model*.

*All notebooks were executed in the Google Collab(because I do not have GPU on my PC), but, unfortunately, they are not reproducible
since I used my Google Disk to repeatedly load my dateset to the notebooks.