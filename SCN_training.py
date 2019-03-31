'''
Semantic Compositional Network https://arxiv.org/pdf/1611.08002.pdf
Developed by Zhe Gan, zg27@duke.edu, July, 12, 2016
'''
import argparse
import json
import sys
import time
import logging
import cPickle

import numpy as np
import scipy.io
import theano
import theano.tensor as tensor

from model_scn.img_cap import init_params, init_tparams, build_model
from model_scn.optimizers import Adam
from model_scn.utils import get_minibatches_idx, zipp, unzip

# Set the random number generators' seeds for consistency
SEED = 123  
np.random.seed(SEED)

OCCURRENCE_DATA = "adjective_noun_occurrence_data"
PAIR_OCCURENCES = "pair_occurrences"

def get_splits_from_occurrences_data(occurrences_data_file, val_set_size=0.0):
    with open(occurrences_data_file, "r") as json_file:
        occurrences_data = json.load(json_file)

    test_images_split = [
        key
        for key, value in occurrences_data[OCCURRENCE_DATA].items()
        if value[PAIR_OCCURENCES] >= 1
    ]

    indices_without_test = [
        key
        for key, value in occurrences_data[OCCURRENCE_DATA].items()
        if value[PAIR_OCCURENCES] == 0
    ]

    train_val_split = int((1 - val_set_size) * len(indices_without_test))
    train_images_split = indices_without_test[:train_val_split]
    val_images_split = indices_without_test[train_val_split:]

    return train_images_split, val_images_split, test_images_split

def get_coco_id_from_path(path):
    return int(path.split("_")[2].split(".")[0])

def prepare_data(seqs):
    
    # x: a list of sentences
    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples)).astype('int64')
    x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask

def calu_negll(f_cost, prepare_data, data, img_feats, tag_feats, iterator):

    totalcost = 0.
    totallen = 0.
    for _, valid_index in iterator:
        x = [data[0][t]for t in valid_index]
        x, mask = prepare_data(x)
        y = np.array([tag_feats[:,data[1][t]]for t in valid_index])
        z = np.array([img_feats[:,data[1][t]]for t in valid_index])
                
        length = np.sum(mask)
        cost = f_cost(x, mask,y,z) * x.shape[1]
        totalcost += cost
        totallen += length
    return totalcost/totallen


""" Training the model. """

def train_model(train, valid, test, img_feats, tag_feats, W, n_words=8791, n_x=300, n_h=512,
    n_f = 512, max_epochs=20, lrate=0.0002, batch_size=64, valid_batch_size=64, 
    dropout_val=0.5, dispFreq=100, validFreq=500, saveFreq=1000,
    saveto = 'coco_result_scn.npz'):
        
    """ n_words : vocabulary size
        n_x : word embedding dimension
        n_h : LSTM/GRU number of hidden units 
        n_f : number of factors
        max_epochs : The maximum number of epoch to run
        lrate : learning rate
        batch_size : batch size during training
        valid_batch_size : The batch size used for validation/test set
        dropout_val : the probability of dropout
        dispFreq : Display to stdout the training progress every N updates
        validFreq : Compute the validation error after this number of update.
        saveFreq : save results after this number of update.
        saveto : where to save.
    """

    options = {}
    options['n_words'] = n_words
    options['n_x'] = n_x
    options['n_h'] = n_h
    options['n_f'] = n_f
    options['max_epochs'] = max_epochs
    options['lrate'] = lrate
    options['batch_size'] = batch_size
    options['valid_batch_size'] = valid_batch_size
    options['dispFreq'] = dispFreq
    options['validFreq'] = validFreq
    options['saveFreq'] = saveFreq
    
    options['n_z'] = img_feats.shape[0]
    options['n_y'] = tag_feats.shape[0]
    options['SEED'] = SEED

    logger.info('Model options {}'.format(options))
    logger.info('{} train examples'.format(len(train[0])))
    logger.info('{} valid examples'.format(len(valid[0])))
    logger.info('{} test examples'.format(len(test[0])))

    logger.info('Building model...')
    
    params = init_params(options,W)
    tparams = init_tparams(params)

    (use_noise, x, mask, y, z, cost) = build_model(tparams,options)
    
    f_cost = theano.function([x, mask, y, z], cost, name='f_cost')
    
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = Adam(tparams, cost, [x, mask, y, z], lr)

    logger.info('Training model...')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)
    
    estop = False  # early stop
    history_negll = []
    best_p = None
    bad_counter = 0    
    uidx = 0  # the number of update done
    start_time = time.time()
    
    try:
        for eidx in xrange(max_epochs):
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(dropout_val)

                x = [train[0][t]for t in train_index]
                y = np.array([tag_feats[:,train[1][t]]for t in train_index])
                z = np.array([img_feats[:,train[1][t]]for t in train_index])
                
                x, mask = prepare_data(x)

                cost = f_grad_shared(x, mask,y,z)
                f_update(lrate)

                if np.isnan(cost) or np.isinf(cost):
                    logger.info('NaN detected')
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    logger.info('Epoch {} Update {} Cost {}'.format(eidx, uidx, cost))
                    
                if np.mod(uidx, saveFreq) == 0:
                    logger.info('Saving ...')
                
                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    np.savez(saveto, history_negll=history_negll, **params)
                    logger.info('Done ...')

                if np.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    
                    #train_negll = calu_negll(f_cost, prepare_data, train, img_feats, kf)
                    valid_negll = calu_negll(f_cost, prepare_data, valid, img_feats, tag_feats, kf_valid)
                    test_negll = calu_negll(f_cost, prepare_data, test, img_feats, tag_feats, kf_test)
                    history_negll.append([valid_negll, test_negll])
                    
                    if (uidx == 0 or
                        valid_negll <= np.array(history_negll)[:,0].min()):
                             
                        best_p = unzip(tparams)
                        bad_counter = 0
                        
                    logger.info('Perp: Valid {} Test {}'.format(np.exp(valid_negll), np.exp(test_negll)))

                    if (len(history_negll) > 10 and
                        valid_negll >= np.array(history_negll)[:-10,0].min()):
                            bad_counter += 1
                            if bad_counter > 10:
                                logger.info('Early Stop!')
                                estop = True
                                break

            if estop:
                break

    except KeyboardInterrupt:
        logger.info('Training interupted')

    end_time = time.time()
    
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)
        
    use_noise.set_value(0.)
    #kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    #train_negll = calu_negll(f_cost, prepare_data, train, img_feats, kf_train_sorted)
    valid_negll = calu_negll(f_cost, prepare_data, valid, img_feats, tag_feats, kf_valid)
    test_negll = calu_negll(f_cost, prepare_data, test, img_feats, tag_feats, kf_test)
    
    logger.info('Final Results...')
    logger.info('Perp: Valid {} Test {}'.format(np.exp(valid_negll), np.exp(test_negll)))
    np.savez(saveto, history_negll=history_negll, **best_p)

    
    logger.info('The code run for {} epochs, with {} sec/epochs'.format(eidx + 1, 
                 (end_time - start_time) / (1. * (eidx + 1))))
    
    return valid_negll, test_negll

def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--occurrences-data",
        help="File containing occurrences statistics about adjective noun pairs",
        required=True,
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == '__main__':

    parsed_args = check_args(sys.argv[1:])
    
    # https://docs.python.org/2/howto/logging-cookbook.html
    logger = logging.getLogger('eval_coco_scn')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('train_coco_scn.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    
    x = cPickle.load(open("./data/coco/data.p","rb"))
    train, val, test = x[0], x[1], x[2]
    wordtoix, ixtoword = x[3], x[4]
    del x
    n_words = len(ixtoword)
    
    x = cPickle.load(open("./data/coco/word2vec.p","rb"))
    W = x[0]
    del x

    train_images_split, val_images_split, test_images_split = get_splits_from_occurrences_data(
        parsed_args.occurrences_data, 0.1
    )

    new_train_0 = []
    new_train_1 = []
    new_train_2 = []
    new_val_0 = []
    new_val_1 = []
    new_val_2 = []
    new_test_0 = []
    new_test_1 = []
    new_test_2 = []

    for i in range(len(train[2])):
        coco_id = unicode(get_coco_id_from_path(train[2][i]))
        if coco_id in train_images_split:
            new_train_0.append(train[0][i])
            new_train_1.append(train[1][i])
            new_train_2.append(train[2][i])
        elif coco_id in val_images_split:
            new_val_0.append(train[0][i])
            new_val_1.append(train[1][i])
            new_val_2.append(train[2][i])
        elif coco_id in test_images_split:
            new_test_0.append(train[0][i])
            new_test_1.append(train[1][i])
            new_test_2.append(train[2][i])

    for i in range(len(val[2])):
        coco_id = unicode(get_coco_id_from_path(val[2][i]))
        if coco_id in train_images_split:
            new_train_0.append(val[0][i])
            new_train_1.append(val[1][i])
            new_train_2.append(val[2][i])
        elif coco_id in val_images_split:
            new_val_0.append(val[0][i])
            new_val_1.append(val[1][i])
            new_val_2.append(val[2][i])
        elif coco_id in test_images_split:
            new_test_0.append(val[0][i])
            new_test_1.append(val[1][i])
            new_test_2.append(val[2][i])

    for i in range(len(test[2])):
        coco_id = unicode(get_coco_id_from_path(test[2][i]))
        if coco_id in train_images_split:
            new_train_0.append(test[0][i])
            new_train_1.append(test[1][i])
            new_train_2.append(test[2][i])
        elif coco_id in val_images_split:
            new_val_0.append(test[0][i])
            new_val_1.append(test[1][i])
            new_val_2.append(test[2][i])
        elif coco_id in test_images_split:
            new_test_0.append(test[0][i])
            new_test_1.append(test[1][i])
            new_test_2.append(test[2][i])

    train = (new_train_0, new_train_1, new_train_2)
    val = (new_val_0, new_val_1, new_val_2)
    test = (new_test_0, new_test_1, new_test_2)

    logger.info('Train set size {}'.format(len(train[0])))
    logger.info('Val set size {}'.format(len(val[0])))
    logger.info('Test set size {}'.format(len(test[0])))


    data = scipy.io.loadmat('./data/coco/resnet_feats.mat')
    img_feats = data['feats'].astype(theano.config.floatX)
    
    data = scipy.io.loadmat('./data/coco/tag_feats.mat')
    tag_feats = data['feats'].astype(theano.config.floatX)

    [val_negll, te_negll] = train_model(train, val, test, img_feats, tag_feats, W,
        n_words=n_words)
        
