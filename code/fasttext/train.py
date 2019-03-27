import fastText
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Read data
data_folder = os.path.join('..', '..', 'data')
all_data = pd.read_csv(os.path.join(data_folder, 'all_data.csv'))

# Grid search through all params
ngrams = list(range(1, 4))
epochs = 100
dims = [20, 300, 700]
lr = [0.01, 0.05, 0.1, 0.2]
runs = []
for ngram in ngrams:
    for d in dims:
        for l in lr:
            runs.append({'dim': d, 'epochs': epochs, 'ngrams': ngram, 'l': l})

num_runs = len(runs)
count = 0
results = {}

for run in runs:
    print('Running {} out of {} parameter sets...'.format(count, num_runs))
    count += 1
    train, test = train_test_split(all_data, test_size=0.2)
    train_path = os.path.join(data_folder, 'train.csv')
    test_path = os.path.join(data_folder, 'test.csv')
    train.to_csv(train_path, header=False, index=False)
    test.to_csv(test_path, header=False, index=False)

    fname = 'fasttext'
    for key, val in run.items():
        fname += '_{}_{}'.format(key, val)
    model_path = os.path.join(data_folder, fname)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    print('Training {}...'.format(fname))
    m = fastText.train_supervised(train_path, lr=run['l'], dim=run['dim'], ws=5, epoch=run['epochs'],
            minCount=1, minCountLabel=0, minn=0, maxn=0, neg=5, wordNgrams=run['ngrams'], loss='softmax',
            bucket=2000000, thread=12, lrUpdateRate=100, t=0.0001, label='__label__', verbose=2,
            pretrainedVectors='')
    m.save_model(os.path.join(model_path, fname + '.bin'))

    print('Testing...')
    test = m.test(test_path)

    # quantize
    # m.quantize()
    # m.save_model(os.path.join(model_path, fname + '.ftz'))

    # store
    results[fname] = {}
    results[fname]['precision'] = test[1]
    results[fname]['recall'] = test[2]
    results[fname]['f1'] = 2*(test[1] * test[2])/(test[1] + test[2])
    results[fname]['run'] = run
    print("Precision: {}, Recall: {}, F1: {}".format(test[1], test[2], results[fname]['f1']))
    print('---------------------------------------')

pd.DataFrame(results).transpose().to_csv(os.path.join(data_folder, 'fasttext_results.csv'))
