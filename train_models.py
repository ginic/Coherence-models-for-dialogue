from __future__ import print_function
from future.utils import iteritems
from builtins import dict
from itertools import permutations
from collections import defaultdict
from sklearn.metrics import accuracy_score
from numpy.random import shuffle
from sklearn import metrics
import tqdm
import argparse
import logging
import numpy as np
import pandas as pd
import os
import errno

from coherence_models_for_dialogue.load_grids import GridLoader
from coherence_models_for_dialogue.corpus.Switchboard.Switchboard import Switchboard
from coherence_models_for_dialogue.generate_grid import corpora_paths, get_corpus
from coherence_models_for_dialogue.generate_shuffled import GridShuffler


# VP added: task names
REORDERING='reordering'
INSERTION='insertion'
CAST='cast'

class EntitiesFeatureExtractor(object):

    #VP: Tweaked params to allow for passing grids directly to object
    def __init__(self, grid_loader=None, grid_folder=None, grids=None, grid_params=None, shuffle_off=False):
        if grids and grid_params:
            self.grids = grids
            self.grids_params = grid_params    
        elif grid_loader:
            self.grid_loader = grid_loader
        else:
            try:
                assert os.path.exists(grid_folder)
            except AssertionError:
                print("The folder " + grid_folder + " does not exist.")
                exit(1)
            self.grid_loader = GridLoader(grid_folder)

        if not self.grids:
            self.grids, self.grids_params = self.grid_loader.get_data()
        
        self.vocabulary = self.get_vocabulary()

        if shuffle_off:
            self.grid_shuffler=None
        else:
            self.grid_shuffler = GridShuffler(grid_folder=grid_folder, grid_loader=grid_loader)

    def update_grids_dct(self, grid_names):
        self.grids = {k:v for k,v in iteritems(self.grids) if k in grid_names}


    def get_transitions_probs_for_grid(self, trans2id, grid_i, transitions_count, transition_range, logprobs=True):
        dummy_value = float(10 ** -10)
        transition_len = transition_range[1]
        transitions_grid_i = self.count_transitions_in_grid_fast(grid_i, transition_len, trans2id)
        probs_grid_i = self.get_probs_normalized_by_trans_of_len(transitions_grid_i, transitions_count)

        # print('N-Transitions range: ', transitions_count)
        # freq_transitions_count = {k: v for k, v in transitions_grid_i.iteritems() if v > 0}
        # print('Transitions count: ', freq_transitions_count)
        # print('Probabilities: ')
        # print(self.show_probs({k: v for k, v in probs_grid_i.iteritems() if k in freq_transitions_count}))

        if logprobs is True:
            logprobs_grid_i = {k: (np.log(v) if v > 0 else np.log(dummy_value)) for k, v in iteritems(probs_grid_i)}
            # print('Log Probabilities: ')
            # print(self.show_probs({k: v for k, v in logprobs_grid_i.iteritems() if k in freq_transitions_count}))
            return logprobs_grid_i
        else:
            return probs_grid_i

    # VP: For CAsT, use this but without shuffling
    def extract_transitions_probs(self,
                                  corpus_dct=None,
                                  transition_range=(2, 2),
                                  saliency=1,
                                  logprobs=True,
                                  corpus_name='Switchboard',
                                  task='reordering'):
        ''' Returns a dict with structure
        {grid_name: [orig_probs, perm1_probs, perm2_probs]}
        where orig_probs is a dict '''

        # TODO: Add <sod> and <eod> tokens?

        print('Params: ')
        print(self.grids_params)
        print('Vocab: ', self.vocabulary)
        print('Vocab size: ', len(self.vocabulary))

        all_combs = self.generate_combinations(self.vocabulary, transition_range)
        # all_combs_str = [''.join(tag for tag in comb) for comb in all_combs]
        all_combs_str = [tuple(tag for tag in comb) for comb in all_combs]
        trans2id = {x:i for i, x in enumerate(all_combs_str)}
        grids_transitions_dict = {}

        # grid_names = ['sw_0001_4325.utt', 'sw_0002_4330.utt', 'sw_0003_4103.utt'] # Testing mode
        # grid_names = ['sw_0755_3018.utt']
        if task != CAST:
            grid_names = [n for n in corpus_dct.keys() if n!='.DS_Store'] #[:10] # Testing mode
            self.update_grids_dct(grid_names)
        else: 
            grid_names = list(self.grids.keys())

        # print('Grids keys : ', grid_names)
        # permuted_files = self.grid_shuffler.generate_shuffled_grids(corpus_dct=corpus_dct, only_grids=grid_names, df=True)

        if task==REORDERING:
            permuted_files = self.grid_shuffler.generate_shuffled_grids(corpus_dct=corpus_dct, only_grids=grid_names,
                                                                        corpus_name=corpus_name,
                                                                        saliency=saliency, df=False)
        elif task==INSERTION:
            permuted_files = self.grid_shuffler.generate_grids_for_insertion(corpus_dct=corpus_dct,
                                                                             only_grids=grid_names,
                                                                             corpus_name=corpus_name,
                                                                             saliency=saliency, df=False)
        else:
            # VP: Just use single original dataframe for Cast
            permuted_files = {k:[v] for (k,v) in self.grids.items()}

        print('Permutation files len: ', len(permuted_files))
        print('First permut len: ', len(permuted_files[grid_names[0]]))
        print('First permut example shape: ', permuted_files[grid_names[0]][0].shape)

        # Compute probs per grid
        for grid_i_name in tqdm.tqdm(grid_names):
            # print('Grid id: ', grid_i_name)

            # Original order for grid_i
            grid_i = self.grids.get(grid_i_name)

            # Check saliency and modify grid accordingly (permuted files were already generated according to saliency)
            if saliency>1:
                grid_i.drop([col for col in grid_i if len([i for i in grid_i[col] if i != '_']) < saliency], axis=1)

            # # Short example
            # grid_i = pd.DataFrame({i: (grid_i[i][:6]) for ind, i in enumerate(grid_i) if ind < 15})
            # print(grid_i)

            permutations_i = permuted_files[grid_i_name]
            transitions_count = self.get_total_numb_trans_given(grid_i, transition_range)

            # The first probs distribution in probs_grid_i is the original one
            if task==REORDERING:
                grids_transitions_dict[grid_i_name] = [self.get_transitions_probs_for_grid(trans2id, grid_ij,
                                                                                           transitions_count, transition_range,
                                                                                           logprobs=logprobs)
                                                       for grid_ij in [grid_i]+permutations_i]
            elif task==INSERTION:
                original = [self.get_transitions_probs_for_grid(trans2id, grid_i, transitions_count,
                                                               transition_range, logprobs=logprobs)]
                reinserted = [[self.get_transitions_probs_for_grid(trans2id, grid_jy,
                                                                  transitions_count, transition_range,logprobs=logprobs) for grid_jy in sent_ind_j]
                                                       for sent_ind_j in permutations_i]
                grids_transitions_dict[grid_i_name] = original+reinserted # TODO: for reinsertion we need one iter per turn removed
            elif task==CAST:
                grids_transitions_dict[grid_i_name] = [self.get_transitions_probs_for_grid(trans2id, grid_i, transitions_count,
                                                               transition_range, logprobs=logprobs)]
                
            self.grids.pop(grid_i_name)

        return grids_transitions_dict


    def show_probs(self, probs_grid_i, top=30):
        for i, x in enumerate(sorted(probs_grid_i, key=lambda x: probs_grid_i[x])):
            print(probs_grid_i[x], " : ", x)
            if i==top:
                break

    def get_total_numb_trans_given(self, grid_i, transition_range):
        n, m = grid_i.shape
        transition_len = transition_range[1]
        transitions_number_per_entity = n-(transition_len-1)
        transitions_number = transitions_number_per_entity*m
        # print('Column length: ', n, " 'Columns number: ", m)
        # print('Total number of transitions of length ', transition_len, " : ", transitions_number)
        return transitions_number

    def get_column_headers(self, grid):
        return list(grid.columns.values)

    def get_probs_normalized_by_trans_of_len(self, transitions_in_grid, transitions_count):
        # 1 - Get probability normalizing per grid counts / number of transitions of len=transition range
        return {comb: np.divide(float(count), float(transitions_count)) for comb, count in iteritems(transitions_in_grid)}


    def get_vocabulary(self):
        return set([role for grid_i in self.grids.values() for entity in grid_i for role in grid_i[entity]])

    def generate_combinations(self, voc, transition_range):
        min_transition, max_transition = transition_range
        doubles = [(tag, tag) for tag in voc]
        all_combs = [comb for i in range(min_transition, max_transition+1) for comb in permutations(voc, i)] + doubles
        return all_combs


    def count_transitions_in_grid_per_entity(self, all_combs_str, grid, saliency=1):
        ''' Returns a list of dictionaries, where a dictionary contains transitions counts for one entity '''

        return [{comb: ''.join(x for x in grid[entity_col]).count(comb) for comb in all_combs_str}
                                            for entity_col in grid]


    def count_transitions_in_grid(self, all_combs_str, grid):
        ''' Returns a list of dictionaries, where a dictionary contains transitions counts for one entity '''

        # all_entities_transitions_in_grid = {comb: ''.join(x for x in grid[entity_col]).count(comb) for comb in all_combs_str
        #                                     for entity_col in grid}
        all_entities_transitions_in_grid = {comb: 0 for comb in all_combs_str}
        for entity_col in grid:
            entity_seq = ''.join(x for x in grid[entity_col])
            for comb in all_combs_str:
                all_entities_transitions_in_grid[comb] = entity_seq.count(comb) + all_entities_transitions_in_grid[comb]
        # print("Len combs: ", len(all_entities_transitions_in_grid))
        return all_entities_transitions_in_grid

    def count_transitions_in_grid_fast_old(self, all_combs_str, grid, trans_range):
        ''' Returns a list of dictionaries, where a dictionary contains transitions counts for one entity '''

        # all_entities_transitions_in_grid = {comb: ''.join(x for x in grid[entity_col]).count(comb) for comb in all_combs_str
        #                                     for entity_col in grid}
        # all_entities_transitions_in_grid = {comb: 0 for comb in all_combs_str}
        n, _ = grid.shape # Column len, number
        count_comb = {comb: 0 for comb in all_combs_str}
        for j in grid:
            tmp = grid[j].tolist()
            for i in range(n - trans_range + 1):
                count_comb[tuple(tmp[i:i + trans_range])] += 1
        return count_comb

    def count_transitions_in_grid_fast(self, grid, trans_range, trans2id):
        ''' Returns a list of dictionaries, where a dictionary contains transitions counts for one entity '''

        # all_entities_transitions_in_grid = {comb: ''.join(x for x in grid[entity_col]).count(comb) for comb in all_combs_str
        #                                     for entity_col in grid}
        # all_entities_transitions_in_grid = {comb: 0 for comb in all_combs_str}

        # trans2id = {('S','O'):1, ('-','-'):2, ...}
        n, _ = grid.shape  # Column len, number
        count_comb = defaultdict(int)
        tmp = grid.T.values.tolist()
        for ent in tmp:
            for i in range(n - trans_range + 1):
                count_comb[trans2id[tuple(ent[i:i + trans_range])]] += 1
        return count_comb


    def featurize_transitions_dct(self, transitions_dict):
        X, y, blocks = [], [], []
        block_i = 0
        for grid_i_name, grids_i in iteritems(transitions_dict):
            # print('Grid name: ', grid_i_name)

            for j, grid_ij in enumerate(grids_i):
                X.append(np.asarray([grid_ij[k] for k in sorted(grid_ij.keys())]))
                y_ij = 0 if j==0 else 1
                y.append(y_ij)
                blocks.append(block_i)
            block_i += 1

        return np.asarray(X), np.asarray(y), np.asarray(blocks)

    def featurize_transitions_dct_svmlightformat_py(self, transitions_dict):
        # (<label>, [(<feature>, <value>), ...], <queryid>)
        np.random.seed(0)
        data = []
        query_i = 0
        for grid_i_name, grids_i in iteritems(transitions_dict):
            # print('Grid name: ', grid_i_name)

            for j, grid_ij in enumerate(grids_i):
                features = [(ind, grid_ij[k]) for ind, k in enumerate(sorted(grid_ij.keys()))]
                label = 1 if j==0 else -1 # 1:coherent, -1:non-coherent
                data.append((label, features, query_i))

            query_i += 1

        shuffle(data)
        return data

    # TODO Follow this example for producing SVMLight format output
    def featurize_transitions_dct_svmlightformat(self, transitions_dict, outfile):
        # # query 1
        # 3 qid:1 1:1 2:1 3:0 4:0.2 5:0
        np.random.seed(0)
        orig_permut_range = list(range(len(list(transitions_dict.values())[0])))
        print('Orig permut range: ', orig_permut_range)
        query_i = 1
        with open(outfile+'.dat', 'w') as to_write:
            for grid_i_name, grids_i in iteritems(transitions_dict):
                # print('Grid name: ', grid_i_name)
                # data_i = []
                to_write.write('# query ' + str(grid_i_name) + '\n')

                # Random order of examples
                np.random.shuffle(orig_permut_range)

                # for j, grid_ij in enumerate(grids_i):
                for j in orig_permut_range:
                    grid_ij = grids_i[j]
                    if j==0:
                        label = 2 # original
                    else:
                        label = 1 # permuted
                    features = [(k+1, grid_ij[k]) for k in sorted(grid_ij.keys())]

                    # data_i.append((label, features, query_i))
                    to_write.write(str(label)+ " qid:"+ str(query_i))
                    for feat_ind, feat_val in features:
                        to_write.write(" "+str(feat_ind) + ":" + str(feat_val))
                    to_write.write('\n')

                query_i += 1

        return

    def featurize_transitions_dct_svmlightformat_insertion(self, transitions_dict, outfile):
        # # query 1
        # 3 qid:1 1:1 2:1 3:0 4:0.2 5:0
        np.random.seed(0)
        # orig_permut_range = range(len(transitions_dict.values()[0]))

        query_i = 1
        with open(outfile+'.dat', 'w') as to_write:
            for grid_i_name, grids_i in iteritems(transitions_dict):
                # print('Grid name: ', grid_i_name)
                # print('Len grids: ', len(grids_i))
                # print('Range permuts: ', range(1, len(grids_i)))

                to_write.write('# query ' + str(grid_i_name) + '\n')


                # for j, grid_ij in enumerate(grids_i):
                for sent_y_ind in range(1, len(grids_i)):

                    sent_y_grids = grids_i[sent_y_ind]
                    y_range = range(1, len(sent_y_grids)+1)
                    y_range.insert(np.random.randint(len(sent_y_grids)), 0) # Randomly insert 0 (original) in y_range

                    # print('Sent y: ', sent_y_ind)
                    # print('Len permuts y: ', len(sent_y_grids))

                    for j in y_range:

                        if j==0:
                            label = 2 # original
                            grid_ij = grids_i[0]
                        else:
                            label = 1 # permuted
                            grid_ij = sent_y_grids[j-1]
                        features = [(k + 1, grid_ij[k]) for k in sorted(grid_ij.keys())]

                        # data_i.append((label, features, query_i))
                        to_write.write(str(label) + " qid:" + str(query_i))
                        for feat_ind, feat_val in features:
                            to_write.write(" "+str(feat_ind) + ":" + str(feat_val))
                        to_write.write('\n')

                    query_i += 1

                # break

        return


def get_grids_transitions_data(grids_transitions_dict, experiments_split, data_type, corpus_name):
    if corpus_name=='Oasis':

        grids_transitions_data = {grid_x_name:val for grid_x_name, val in iteritems(grids_transitions_dict)
                                  if grid_x_name in experiments_split[data_type].tolist()}
    elif corpus_name=='AMI':

        grids_transitions_data = {grid_x_name:val for grid_x_name, val in iteritems(grids_transitions_dict)
                                  if grid_x_name[:-1] in experiments_split[data_type].tolist()}
    else:
        grids_transitions_data = {grid_x_name:val for grid_x_name, val in iteritems(grids_transitions_dict)
                                  if grid_x_name in experiments_split[data_type].tolist()}

    return grids_transitions_data


def parse():
    parser = argparse.ArgumentParser(description='Feature vectors generator')
    parser.add_argument('-g', '--generate_feature_vectors', default='Oasis', help='Generate feature vectors')
    parser.add_argument('-m', '--grid_mode', default='egrid_-coref', help='Grid mode')
    parser.add_argument('-ta', '--task', default='reordering',
                        help='Task type')  # possible values: reordering, insertion
    parser.add_argument('-sa', '--saliency', default=1, help='Saliency')
    parser.add_argument('-nt', '--number_transitions', default=(2, 2), help='Transition range')
    args = parser.parse_args()
    return args


def run(args):
    corpus_name, grid_mode, task_type, saliency, number_transitions = \
        args.generate_feature_vectors, args.grid_mode, args.task, \
        args.saliency, args.number_transitions

    if args.generate_feature_vectors:
        print(''.join(y for y in ["-"] * 180))

        corpus = corpus_name
        model_type = grid_mode
        saliency = saliency
        transition_range = number_transitions
        task = task_type
        only_data = ['training', 'test', 'validation']
        # only_data = ['test','validation']
        # only_data = ['training']

        ########################
        data_types = {'training': 'train', 'test': 'test', 'validation': 'dev'}
        experiments_path = 'experiments/'
        grids_data_path = 'data/'

        out_path = create_path(experiments_path + corpus + '/' + task + '/' + model_type + '/')
        grids_path = grids_data_path + corpus + '/' + model_type + '/'
        data_filename = grids_path.split('/')[1] + '_sal' + str(saliency) + '_range' + str(transition_range[0]) + "_" + str(
            transition_range[1])

        print('Grid folder: ', grids_path)
        grid_loader = GridLoader(grids_path)

        corpus_dct, corpus_loader = get_corpus(corpus)

        # Get train val test splits
        experiments_split = grid_loader.get_training_test_splits(corpus_name=corpus)  # type pd.Dataframe
        print('Train Test split', experiments_split.shape)
        print('Training data', len(experiments_split['training']))
        print('Test data', len(experiments_split['test']))

        # Reduce data
        # corpus_dct = {k:corpus_dct[k] for i, k in enumerate(corpus_dct.keys()) if i==0} # Testing


        if corpus == 'AMI':
            selected_files_list = list(
                set([grid_name + '.' for data in only_data for grid_name in experiments_split[data].tolist()]))
        else:
            selected_files_list = list(set([grid_name for data in only_data for grid_name in experiments_split[data].tolist()]))

        corpus_dct = {k: corpus_dct[k] for k in corpus_dct.keys() if k in selected_files_list}

        feature_extractor = EntitiesFeatureExtractor(grid_folder=grids_path, grid_loader=grid_loader)

        print('Corpus name: ', corpus)
        print('Length selected files: ', len(selected_files_list))
        print('Model type: ', model_type)
        print('Data type: ', only_data)
        print('Task: ', task)
        print('Saliency: ', saliency)
        print('Transition_range: ', transition_range)
        grids_transitions_dict = feature_extractor.extract_transitions_probs(corpus_dct=corpus_dct,
                                                                             transition_range=transition_range,
                                                                             saliency=saliency,
                                                                             logprobs=True,
                                                                             corpus_name=corpus,
                                                                             task=task)

        print('Grid trans dct len: ', len(grids_transitions_dict))
        print('Grid trans key example: ', list(grids_transitions_dict.keys())[0])
        # print('Grid trans dct len: ', grids_transitions_dict.keys())

        for data_type in only_data:

            filename = data_types[data_type]

            grids_transitions_test = get_grids_transitions_data(grids_transitions_dict,
                                                                experiments_split, data_type, corpus)
            if task == 'reordering':
                feature_extractor.featurize_transitions_dct_svmlightformat(grids_transitions_test,
                                                                           out_path + data_filename + '_' + filename)
            elif task == 'insertion':
                feature_extractor.featurize_transitions_dct_svmlightformat_insertion(grids_transitions_test,
                                                                                     out_path + data_filename + '_' + filename)

def create_path(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    return filename

if __name__ == '__main__':
    # main()
    args = parse()
    run(args)