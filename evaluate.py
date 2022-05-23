import os
from regex import W
import torch 
import torch.nn.functional as F
import pickle 
from data import Corpus, Dictionary

def critical_region(language):
    dict_conditions = dict()
    if language == 'EU':
        dict_conditions['1'] = 5
        dict_conditions['2'] = 5
        dict_conditions['3'] = 5 
        dict_conditions['4'] = 5
    elif language == 'HI1':
        dict_conditions['AMI'] = 1
        dict_conditions['AFI'] = 1
        dict_conditions['AMP'] = 1
        dict_conditions['AFP'] = 1
        dict_conditions['UMI'] = 2
        dict_conditions['UFI'] = 2
        dict_conditions['UMP'] = 2
        dict_conditions['UFP'] = 2
    elif language == 'HI2':
        dict_conditions['A'] = 2 # Unamb P IPFV
        dict_conditions['B'] = 1  # Amb P IPFV
        dict_conditions['C'] = 1  # Amb A IPFV
        dict_conditions['D'] = 1  # Amb S IPFV
        dict_conditions['E'] = 2  # Unamb P PFV
        dict_conditions['F'] = 2 # Unamb A PFV
        dict_conditions['G'] = 1 # Amb P PFV
        dict_conditions['H'] = 1 # Amb S PFV

    elif language == 'DE2':
        dict_conditions['1'] = 9
        dict_conditions['2'] = 9
        dict_conditions['3'] = 9
        dict_conditions['4'] = 9
        dict_conditions['5'] = 9
        dict_conditions['6'] = 9
    elif language == 'DE1':
        dict_conditions['acc_subject'] = 0 
        dict_conditions['dat_subject'] = 0
        dict_conditions['acc_object'] = 0
        dict_conditions['dat_object'] = 0

    else: 
        print('wrong langauge indicated')
    return dict_conditions

def check_oov_de1(sent, dictionary):
    oov = False
    sent = sent[-6:]
    list_oovs = []
    for i,t in enumerate(sent):
        #print(t, dictionary.word2idx['<unk>'])
        if t == dictionary.word2idx['<unk>']:
            oov = True
            list_oovs.append(i)
        else:
            continue
    return oov, list_oovs

def check_oov_de2(sent, dictionary):
    pass

def check_oov(sent, id_oov, dictionary):
        oov = False
        list_oovs = []
        for i,t in enumerate(sent[:id_oov+1]):
            if t == dictionary.word2idx['<unk>']:
                oov = True
                list_oovs.append(i)
            else:
                continue
        return oov, list_oovs
    

def read_stimuli(path, dictionary, max_l):
    stimuli = []

    with open(path, 'r') as rf:
        next(rf)
        for l in rf:
            l = l.strip().split('\t')
            id = l[0]
            cond = l[1]
            sent = l[2].replace(',', ' ,')
            sent = sent.replace('.', ' .')
            sent = sent.split()
            char_input = []
            word_target = []
            for word in sent:
                word = word.strip().lower()
                try:
                    word_target.append(dictionary.word2idx[word])
                except:
                    print('oov: ', word)
                    word_target.append(dictionary.word2idx['<unk>'])
                word = []
                char_tens = torch.zeros(max_l)
                for i,c in enumerate(word):
                    char_tens[i+1] = dictionary.char2idx[c] 
                char_tens[0] = dictionary.char2idx['<bow>']
                char_input.append(char_tens)
            stimuli.append((id, cond, char_input, word_target))
    return stimuli
    

def get_dictionary(path):
    sl = 20
    f = open(path, 'rb')
    corpus = pickle.load(f)
    f.close()
    return corpus 

def get_surprisal(model, stimuli,dictionary, lang, oov=True):
    surps = dict()
    dict_cond = critical_region(lang)
    for nr, s in enumerate(stimuli):
        char_input = s[2]
        char_input = torch.stack(char_input).long()
        char_input = char_input.unsqueeze(0)
        word_target = s[3]
        outofvoc = False
        if oov:
            if lang == 'DE1':
                outofvoc, list_oovs = check_oov_de1(word_target, dictionary)
            elif lang == 'DE2':
                outofvoc = check_oov_de2()
            else:
                outofvoc, list_oovs = check_oov(word_target, dict_cond[s[1]], dictionary)

        if not outofvoc:
            out = model(char_input, evaluate=True).squeeze(0)
            suppi = []
            print('char out', char_input.shape, out.shape)
            for i in range(0, (out.shape[0])):
                #print(out[i][word_target[i]])
                suppi.append((-out[i][word_target[i]], dictionary.idx2word[word_target[i]]))
            #print(out.shape)
            print(len(suppi))
            surps[nr] = {'id':s[0], 'cond':s[1], 'surp': suppi}
        #for i, word_idx in enumerate(out):
        #    surps.append(out[i][word_target[i]])
    return surps 

def write_to_file(surprisals, path):
    with open(path, 'w') as wf:
        wf.write('{}\t{}\t{}\t{}\t{}\n'.format('id', 'cond','s_id','word', 'surprisal'))
        for k,v in surprisals.items():
            surps = v['surp']
            for i,s in enumerate(surps):
                sur = s[0]
                word = s[1]
                wf.write('{}\t{}\t{}\t{}\t{}\n'.format(v['id'], v['cond'], str(i), word, sur))




if __name__ == '__main__':
    lang = 'EU'
    stimuli_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/agent_lms/stimuli/Basque_psych_LSTM.csv'
    model_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/CharCNN_Pytorch/trained_models/Basque_CNN2/model'
    out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/agent_lms/results/CNN_basque.txt'
    data_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/training_data_lstms/EU/chardata'
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    model = torch.load(model_path, map_location=map_location)
    corpus = get_dictionary(data_path)
    stim = read_stimuli(stimuli_path, corpus.dictionary, corpus.max_l)
    surps = get_surprisal(model, stim, corpus.dictionary, lang)
    write_to_file(surps, out_path)