from transformers import BartTokenizer
import codecs
import os
import numpy as np
import h5py
import data_loading_helpers_modified as dh
from glob import glob
from tqdm import tqdm
import pickle
import torch
from copy import deepcopy


def check_nan_inf(input_data,text):
    # 检查是否存在NaN
    # print("enter")
    nan_check = torch.isnan(input_data)
    if nan_check.any().item():
        print(text,"中输入数据中存在NaN")

    # 检查是否存在Inf
    inf_check = torch.isinf(input_data)
    if inf_check.any().item():
        print(text,"中输入数据中存在Inf")

def normalize_1d(input_tensor):
    # normalize a 1d tensor
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean) / std
    return input_tensor



def normalize_2d(input_matrix):
    # 将矩阵展平为1D张量
    flattened_tensor = input_matrix.view(-1)

    # 计算整个矩阵的均值和标准差
    mean = flattened_tensor.mean()
    std = flattened_tensor.std()

    # 对整个矩阵进行标准化
    normalized_matrix = (input_matrix - mean) / std

    return normalized_matrix


def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands, dim):
    frequency_features = []
    for band in bands:
        frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type + band][0:dim])
    word_eeg_embedding = np.concatenate(frequency_features)
    if len(word_eeg_embedding) != dim * len(bands):
        print(
            f'expect word eeg embedding dim to be {dim * len(bands)}, but got {len(word_eeg_embedding)}, return None')
        return None, None
    assert len(word_eeg_embedding) == dim * len(bands)
    return_tensor = torch.from_numpy(word_eeg_embedding)

    return normalize_1d(return_tensor), return_tensor

# sentence级别的eeg embedding
def get_sent_eeg(sent_obj, bands):
    sent_eeg_features = []
    # 特征只使用了mean_t1,将八个频段的mean找出来，并进行拼接
    for band in bands:
        key = 'mean' + band
        sent_eeg_features.append(sent_obj['sentence_level_EEG'][key])
    sent_eeg_embedding = np.concatenate(sent_eeg_features)
    assert len(sent_eeg_embedding) == 105 * len(bands)
    return_tensor = torch.from_numpy(sent_eeg_embedding)
    normalize_1d_return_tensor = normalize_1d(return_tensor)
    # 归一化
    return normalize_1d_return_tensor,return_tensor

def get_input_sample(sent_obj, tokenizer, eeg_type='GD', bands=['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'],
                     max_len=58, dim=105, add_CLS_token=False):
    if sent_obj is None:
        return None

    input_sample = {}

    target_string = sent_obj['content']
    # handle some wierd cases
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty', 'empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1', 'film.')

    # https://github.com/huggingface/transformers/blob/f85acb4d73a84fe9bee5279068b0430fc391fb36/src/transformers/tokenization_utils_base.py#L2852
    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, truncation=True,
                                 return_tensors='pt', return_attention_mask=True)
    input_sample['target_tokenized'] = target_tokenized
    input_sample['target_ids'] = target_tokenized['input_ids'][0]

    # get sentence level EEG features
    sent_level_eeg_tensor,non_normalized_sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        # print('[NaN sent level eeg]: ', target_string)
        return None
    """
    sent_level_eeg_tensor = torch.nan_to_num(sent_level_eeg_tensor, nan=1e-5)
    non_normalized_sent_level_eeg_tensor = torch.nan_to_num(non_normalized_sent_level_eeg_tensor,nan=1e-5)
    """


    word_embeddings = []
    non_normalized_word_embeddings = []
    selected_words = []

    if len(sent_obj['word']) == 0:
        return None

    for word in sent_obj['word']:
        word_level_eeg_tensor, non_normalized_word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands=bands,
                                                                                            dim=dim)
        if word_level_eeg_tensor is None:
            return None
        if torch.isnan(word_level_eeg_tensor).any():
            return None

        word_embeddings.append(word_level_eeg_tensor)
        non_normalized_word_embeddings.append(non_normalized_word_level_eeg_tensor)

        selected_words.append(word["content"])

    # """ensure not to exceed to max-length"""
    # if len(word_embeddings) > max_len:
    #     word_embeddings = word_embeddings[:max_len]
    #     non_normalized_word_embeddings = non_normalized_word_embeddings[:max_len]
    #     selected_words = selected_words[:max_len]

    # word_sentence embedding
    word_embeddings.append(sent_level_eeg_tensor)
    non_normalized_word_embeddings.append(non_normalized_sent_level_eeg_tensor)

    """for visulization"""
    input_sample["embeddings_for_vis"] = deepcopy(word_embeddings)
    input_sample["non_normalized_embeddings_for_vis"] = deepcopy(non_normalized_word_embeddings)
    input_sample["selected_words"] = selected_words


    non_normalized_word_embeddings = torch.stack(non_normalized_word_embeddings)
    normalized_word_sentence_embeddings = normalize_2d(non_normalized_word_embeddings)
    normalized_word_sentence_embeddings = list(torch.unbind(normalized_word_sentence_embeddings))

    """get true sequence length"""
    seq_len = len(word_embeddings)


    # pad to max_len
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(dim * len(bands)))
        normalized_word_sentence_embeddings.append(torch.zeros(dim * len(bands)))

    input_sample['input_embeddings'] = torch.stack(word_embeddings)  # max_len * (105*num_bands)
    input_sample['normalized_input_embeddings'] = torch.stack(
        normalized_word_sentence_embeddings)  # max_len * (105*num_bands)

    # mask out padding tokens
    input_sample['input_attn_mask'] = torch.zeros(max_len)  # 0 is masked out

    #TODO: Feng的做法
    input_sample['input_attn_mask'][:seq_len] = torch.ones(seq_len)  # 1 is not masked
    #TODO:Wang的做法
    # input_sample['input_attn_mask'][:len(sent_obj['word'])] = torch.ones(len(sent_obj['word']))# 1 is not masked

    # mask out padding tokens reverted: handle different use case: this is for pytorch transformers
    input_sample['input_attn_mask_invert'] = torch.ones(max_len)  # 1 is masked out

    #TODO: Feng的做法
    input_sample['input_attn_mask_invert'][:seq_len] = torch.zeros(seq_len)  # 0 is not masked
    #TODO:Wang的做法
    # input_sample['input_attn_mask_invert'][:len(sent_obj['word'])] = torch.zeros(len(sent_obj['word']))  # 0 is not masked

    # mask out target padding for computing cross entropy loss
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = target_string.split()  # maybe bug but we don't use this item
    input_sample['target_string'] = target_string



    # 这里是有fixation的词,每个词本来是对应了一个EEG段落，这里是为了构建CLIP
    target_string_has_fixation = sent_obj['word_tokens_has_fixation']

    real_word_token = []
    if len(target_string_has_fixation) == 0:
        return None
    for word in target_string_has_fixation:
        word = word.lower().replace('.', '')
        word = word.replace(',', '')
        word = word.replace('!', '')
        word = word.replace('?', '')
        word = word.replace(';', '')
        word = word.replace(':', '')
        word = word.replace('"', '')
        word = word.replace('(', '')
        word = word.replace(')', '')
        word = word.replace("'", '')
        word_token = tokenizer([word], padding='max_length', max_length=max_len,
                                    truncation=True, return_attention_mask=True, return_length=True,
                                    is_split_into_words=True)
        word_idx = word_token['input_ids']
        if tokenizer.eos_token_id is not None and tokenizer.bos_token_id is not None:
            if ((word_idx.index(tokenizer.eos_token_id) - word_idx.index(tokenizer.bos_token_id)) == 2):
                real_word_token.append(word_idx[1])
            else:  # 意思是1个词经过BPE之后得到了多个token, 这种情况在BELT里面需要跳过,才方便做contrastive learning
                a = tokenizer.pad_token_id
                real_word_token.append(tokenizer.pad_token_id)
                # pad#valid_word_token.append('<pad>')
        else:  # 如果tokenizer没有定义bos和eos,那么就不需要跳过
            print("tokenizer has no bos and eos")
            raise NotImplementedError


    # target_string_has_fixation =valid_word_list
    # target_tokenized_has_fixation = tokenizer(target_string_has_fixation,padding='max_length',max_length=max_len,truncation=True,return_attention_mask=True,is_split_into_words=True)
    input_sample['target_id_has_fixation'] = real_word_token

    # 相当于是不使用这个targetmask

    input_sample['seq_len_has_fixation'] = len(
        sent_obj['word_tokens_has_fixation'])
    try:
        assert len(input_sample['target_id_has_fixation']
                   ) == input_sample['seq_len_has_fixation']
    except:
        print(target_string_has_fixation, input_sample['seq_len_has_fixation'], len(
            target_string_has_fixation))
        return None
    while len(input_sample['target_id_has_fixation']) < max_len:
        input_sample['target_id_has_fixation'].append(1)

    input_sample['word_tokens_has_fixation'] = sent_obj['word_tokens_has_fixation']
    while len(input_sample['word_tokens_has_fixation']) < max_len:
        input_sample['word_tokens_has_fixation'].append("")
    # clean 0 length data
    if input_sample['seq_len'] == 0:
        print('discard length zero instance: ', target_string)
        return None

    return input_sample


if __name__ == "__main__":
    version = 'v2'
    eeg_type = "GD"  # gaze duration (GD)
    max_len = 58
    dim = 105
    bands = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']
    task_name = "task3-TSR-2.0"
    is_add_CLS_token = False

    print("load tokenizer......")
    tokenizer = BartTokenizer.from_pretrained('./models/huggingface/bart-large')

    print(f'start processing ZuCo-{version} {task_name}...')

    # load files
    input_mat_files_dir = f"./zuco_dataset/{task_name}/Matlab_files"
    mat_files = glob(os.path.join(input_mat_files_dir, '*.mat'))
    mat_files = sorted(mat_files)

    dataset_dict = {}

    for mat_file in tqdm(mat_files):

        subject = os.path.basename(mat_file).split('_')[0].replace('results', '').strip()
        if subject != 'YMH':
            assert subject not in dataset_dict
            dataset_dict[subject] = []

            f = h5py.File(mat_file, 'r')
            sentence_data = f['sentenceData']

            # sent level eeg
            # mean_t1 = np.squeeze(f[sentence_data['mean_t1'][0][0]][()])
            mean_t1_objs = sentence_data['mean_t1']
            mean_t2_objs = sentence_data['mean_t2']
            mean_a1_objs = sentence_data['mean_a1']
            mean_a2_objs = sentence_data['mean_a2']
            mean_b1_objs = sentence_data['mean_b1']
            mean_b2_objs = sentence_data['mean_b2']
            mean_g1_objs = sentence_data['mean_g1']
            mean_g2_objs = sentence_data['mean_g2']

            rawData = sentence_data['rawData']
            contentData = sentence_data['content']
            omissionR = sentence_data['omissionRate']
            wordData = sentence_data['word']

            for idx in range(len(rawData)):
                # get sentence string
                obj_reference_content = contentData[idx][0]
                sent_string = dh.load_matlab_string(f[obj_reference_content])

                sent_obj = {'content': sent_string}

                # get sentence level EEG
                sent_obj['sentence_level_EEG'] = {
                    'mean_t1':np.squeeze(f[mean_t1_objs[idx][0]][()]),
                    'mean_t2':np.squeeze(f[mean_t2_objs[idx][0]][()]),
                    'mean_a1':np.squeeze(f[mean_a1_objs[idx][0]][()]),
                    'mean_a2':np.squeeze(f[mean_a2_objs[idx][0]][()]),
                    'mean_b1':np.squeeze(f[mean_b1_objs[idx][0]][()]),
                    'mean_b2':np.squeeze(f[mean_b2_objs[idx][0]][()]),
                    'mean_g1':np.squeeze(f[mean_g1_objs[idx][0]][()]),
                    'mean_g2':np.squeeze(f[mean_g2_objs[idx][0]][()])
                }

                sent_obj['word'] = []

                # get word level data
                word_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask = dh.extract_word_level_data(
                    f, f[wordData[idx][0]])

                if word_data == {}:
                    print(f'missing sent: subj:{subject} content:{sent_string}, append None')
                    dataset_dict[subject].append(None)
                    continue
                elif len(word_tokens_all) == 0:
                    print(f'no word level features: subj:{subject} content:{sent_string}, append None')
                    dataset_dict[subject].append(None)
                    continue

                else:
                    for widx in range(len(word_data)):
                        data_dict = word_data[widx]
                        word_obj = {'content': data_dict['content'], 'nFixations': data_dict['nFix']}
                        if 'GD_EEG' in data_dict:
                            gd = data_dict["GD_EEG"]
                            ffd = data_dict["FFD_EEG"]
                            trt = data_dict["TRT_EEG"]
                            assert len(gd) == len(trt) == len(ffd) == 8
                            word_obj['word_level_EEG'] = {
                                'GD': {'GD_t1': gd[0], 'GD_t2': gd[1], 'GD_a1': gd[2], 'GD_a2': gd[3], 'GD_b1': gd[4],
                                       'GD_b2': gd[5], 'GD_g1': gd[6], 'GD_g2': gd[7]},
                                'FFD': {'FFD_t1': ffd[0], 'FFD_t2': ffd[1], 'FFD_a1': ffd[2], 'FFD_a2': ffd[3],
                                        'FFD_b1': ffd[4], 'FFD_b2': ffd[5], 'FFD_g1': ffd[6], 'FFD_g2': ffd[7]},
                                'TRT': {'TRT_t1': trt[0], 'TRT_t2': trt[1], 'TRT_a1': trt[2], 'TRT_a2': trt[3],
                                        'TRT_b1': trt[4], 'TRT_b2': trt[5], 'TRT_g1': trt[6], 'TRT_g2': trt[7]}
                            }
                            sent_obj['word'].append(word_obj)

                    sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
                    sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
                    sent_obj['word_tokens_all'] = word_tokens_all

                    dataset_dict[subject].append(sent_obj)

            total_num_sentence = len(dataset_dict[subject])
            train_divider = int(0.8 * total_num_sentence)
            dev_divider = train_divider + int(0.1 * total_num_sentence)

            for i in range(train_divider):
                input_sample = get_input_sample(dataset_dict[subject][i], tokenizer, eeg_type,
                                                bands=bands, add_CLS_token=is_add_CLS_token, max_len=max_len, dim=dim)
                if input_sample is not None:
                    output_name = f"./datasets/data_word_sentence_5/train/{version}-{task_name}-{subject}-{i}.pickle"
                    with codecs.open(output_name, 'wb') as handle:
                        pickle.dump(input_sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

            for i in range(train_divider, dev_divider):
                input_sample = get_input_sample(dataset_dict[subject][i], tokenizer, eeg_type,
                                                bands=bands, add_CLS_token=is_add_CLS_token, max_len=max_len, dim=dim)
                if input_sample is not None:
                    output_name = f"./datasets/data_word_sentence_5/valid/{version}-{task_name}-{subject}-{i}.pickle"
                    with codecs.open(output_name, 'wb') as handle:
                        pickle.dump(input_sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

            for i in range(dev_divider, total_num_sentence):
                input_sample = get_input_sample(dataset_dict[subject][i], tokenizer, eeg_type,
                                                bands=bands, add_CLS_token=is_add_CLS_token, max_len=max_len, dim=dim)
                if input_sample is not None:
                    output_name = f"./datasets/data_word_sentence_5/test/{version}-{task_name}-{subject}-{i}.pickle"
                    with codecs.open(output_name, 'wb') as handle:
                        pickle.dump(input_sample, handle, protocol=pickle.HIGHEST_PROTOCOL)



