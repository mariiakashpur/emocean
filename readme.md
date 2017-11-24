# This file explains how to create BOW, OCC and BOW+OCC feature files from datasets, and how to run the main program on them


to generate a csv feature file for ML:BOW model (all 4 corpora examples)
```bash
python generate_csv.py data/aman/Emotion-Data_Aman/Benchmark/category_gold_std.txt DD lexicons/positive-words.txt lexicons/negative-words.txt NEW_aman_DD.csv
python generate_csv.py data/alm DD lexicons/positive-words.txt lexicons/negative-words.txt NEW_alm_DD.csv
python generate_csv.py data/tweets/Saif_Mohammad_tweets.txt DD lexicons/positive-words.txt lexicons/negative-words.txt NEW_tweets_DD.csv
python generate_csv.py data/semeval/1.txt DD lexicons/positive-words.txt lexicons/negative-words.txt NEW_semeval_DD.csv data/semeval/1.xml
```

to generate a csv feature file for ML:OCC_DEP model (example on Alm corpus)
```bash
python generate_csv.py data/alm OCC lexicons/positive-words.txt lexicons/negative-words.txt NEW_alm_OCC.csv
```

to generate a csv feature file for ML:OCC_SRL model (example on Alm corpus)
```bash
python generate_csv.py data/alm OCCSRL lexicons/positive-words.txt lexicons/negative-words.txt NEW_alm_SRL.csv
```

to generate a combined csv feature file for ML:BOW+OCC_DEP model(example on Aman corpus)
```bash
python generate_csv_comb.py NEW_aman_DD.csv NEW_aman_OCC.csv NEW_aman_COMB.csv
```
to run main program for Alm corpus and RB:OCC_DEP model
```bash
python main.py NEW_alm_OCC.csv OCCRB
```
to run main program for Alm corpus and RB:OCC_SRL model
```bash
python main.py NEW_alm_SRL.csv OCCRB
```
to run main program for Alm corpus and ML:OCC_DEP model
```bash
python main.py NEW_alm_OCC.csv OCC
```
to run main program for Alm corpus and ML:OCC_SRL model
```bash
python main.py NEW_alm_SRL.csv OCC
```
to run main program for Alm corpus and ML:BOW model
```bash
python main.py NEW_alm_DD.csv DD
```
to run main program for Alm corpus and ML:BOW+OCC_DEP model
```bash
python main.py NEW_alm_COMB.csv COMB
```
to run main program for Alm corpus and ML:BOW+OCC_SRL model
```bash
python main.py NEW_alm_COMB_SRL.csv COMB
```


# emocean
