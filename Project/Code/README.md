# Climate Misinfomation Detection System

Although there is ample scientific evidence on climate change, there is still a lingering debate over fundamental questions whether global warming is human-induced. One factor that contributes to this continuing debate is that ideologically-driven misinformation is regularly being distributed on mainstream and social media.
The challenge of the project is to build a system that detects whether a document contains climate change misinformation. The task is framed as a binary classification problem: a document either contains climate change misinformation or it doesn’t.

### Environment

- Windows 10 (Python 3.5+)
- Tensorflow 2

### JSON files:

- train.json (used for train, 1168 positive samples)
- balanced.json (used for train, 50 positive samples, 50 negative samples)

- dev.json (used for development and evaluation, 50 positive samples, 50 negative samples)
- dev-baseline-r.json (used for evaluation, a baseline of prediction for `dev.json`)

- test-unlabelled.json (used for prediction)


### Python files:

- Method1.py (Also contains two basic examples [1> how to build Sequential Layers using Keras and 2> how to construct your own train-dev-test dataset to feed the model] to use Tensorflow2)
- Method2.py
- Method3.py
- scoring.py

Each 'MethodX' Python files implemtents one method to solve this project, the details of each method can be found in `Report.pdf`. 

Run `train()` funtion in each file, you will get 2 kinds of results such as: 'test-output_M1_0.5.json' and 'dev-output_M1_0.5.json'. The first one is the prediction of `test-unlabelled.json`, used for compitetion; the second is the prediction of `dev.json`, used for evaluation of the method (using `scoring.py`, such as "python3 scoring.py --groundtruth dev.json --predictions dev-output_M2_0.5.json").
