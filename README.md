# Incongruity-SemEval-2020-Task-7
Final project for [Natural Language Processing (CS 159)](https://sites.google.com/g.hmc.edu/cs159spring2021) at Harvey Mudd College in Spring 2021. We analyzed incongruity in humor based on SemEval 2020 Task 7 and the Humicroedit database.

## GloVe Vectors
Download GloVe vectors from the following link: [http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip). Then put them in a folder called `/glove`.

Code to use GloVe vectors adapted from [https://sites.google.com/g.hmc.edu/cs159spring2021/labs/lab-4-glove-vectors](https://sites.google.com/g.hmc.edu/cs159spring2021/labs/lab-4-glove-vectors).

## Duluth at SemEval 2020 Task 7
We ran the Duluth models using the code at https://github.com/dora-tang/SemEval-2020-Task-7. We used the following versions:

```
Python              3.7.10
torch               1.8.1
torchtext           0.9.1
numpy               1.20.1
pandas              1.2.4
transformers        4.5.1
allennlp            2.4.0
spacy               2.3.5
nltk                3.5
tensorboardX        2.2
```

We also made the following changes to the Duluth code. We replaced `src/util.py` line 31 with:
```
from torchtext.legacy import data
```
We also replaced `src/models.py` line 295:
```
outputs = self.transformer(inp, attention_mask=inp_mask, return_dict = False)
```
