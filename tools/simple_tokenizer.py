# source: https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/simple_tokenizer.py


import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re
import random

import torch
import nltk
from nltk.corpus import wordnet as wn
# import numpy as np
# import spacy

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def init_set(path):
    with open(path) as file:
        lines = [line.rstrip() for line in file]
    return set(lines)

def init_list(path):
    with open(path) as file:
        lines = [line.rstrip() for line in file]
    return lines

def del_list_inplace(l, id_to_del):
    for i in sorted(id_to_del, reverse=True):
        del (l[i])



def synonym_antonym_extractor(phrase, pos):
    synonyms = []
    for syn in wn.synsets(phrase, pos):
        for name in syn.lemma_names():
            synonyms.append(name)
    synonyms = list(set(synonyms))
    return random.choice(synonyms) if synonyms else phrase

class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe(),
                 remove_color=False,
                 drop_pro=(0.5, 0.2, 0.1),
                 on_sentence=False,
                 remove_color_type='link',
                 sen_drop_prob=0.5,
                 random_remove_set=(1., 2),
                 mod_percent_per_sen=0.15,
                 BERT_drop_prob=0.8,
                 BERT_remove=False,
                 BERT_change_prob=0.1,
                 mask_color_type='random_mask_BERT_color',
                 mask_color=False,
                 mask_prob_per_sen=0.15,
                 mask_prob=0.8,
                 per_sen_max=2,
                 encode_type='default',
                 drop_prob_per_sen=0.2,
                 change_sen_prob=0.5,
                 ):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.pop(-1)
        vocab.extend(['<|maskoftext|>', '<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|maskoftext\|>|<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
        self.color_words_high = init_set('tools/colors/high.txt')
        self.color_words_middle = init_set('tools/colors/middle.txt')
        self.color_words_low = init_set('tools/colors/low.txt')
        self.sen_drop_prob = sen_drop_prob
        self.drop_high = drop_pro[0]
        self.drop_middle = drop_pro[1]
        self.drop_low = drop_pro[2]
        self.remove_color = remove_color
        self.on_sentence = on_sentence
        self.per_sen_max = per_sen_max
        self.link = ["and", "with", "-", "/", "or"]
        
        self.random_remove_set = random_remove_set
        self.mod_percent_per_sen = mod_percent_per_sen
        self.BERT_drop_prob = BERT_drop_prob
        self.BERT_remove = BERT_remove
        self.BERT_change_prob = BERT_change_prob
        


        self.color_words = set.union(self.color_words_high, self.color_words_middle, self.color_words_low)
        self.color_words_with_index = {k: v + 1 for v, k in enumerate(self.color_words)}
        self.remove_f = {
            'simple': self.remove_color_simple,
            'link': self.remove_color_with_link,
            'one_per_sen': self.remove_color_one_per_sen,
            'one_sen_prob': self.remove_color_one_sen_prob,
            'one_sen_prob_label': self.remove_color_one_sen_prob_label,
            'one_sen_prob_label_simple': self.remove_color_one_sen_prob_label_simple,
            'multi_sen': self.remove_color_multi_sen,
            'multi_sen_simple': self.remove_color_multi_sen_simple,
            'random_remove_fix': self.random_remove_fix_num,
            'random_remove_max': self.random_remove_max_num,
            'random_remove_BERT': self.random_remove_BERT,
            'random_remove_BERT_replace': self.random_remove_BERT_replace,
            'random_remove_BERT_color': self.random_remove_BERT_color,
        }[remove_color_type]
        self.mask_color = mask_color
        self.mask_color_f ={
            'random_mask_BERT_color': self.random_mask_BERT_color,
            'random_mask_color': self.random_mask_color,
        }[mask_color_type]
        
        self.encode = {
            'default': self.encode_default,
            'mask_nouns': self.encode_mask_nouns,
            'nouns_adjs_drop': self.encode_nouns_adjs_drop,
            'adjs_drop': self.encode_adjs_drop,
            'nouns_drop': self.encode_nouns_drop,
            'nouns_adjs_verb_drop': self.encode_nouns_adjs_verb_drop,
            'BERT_like_drop': self.encode_BERT_like_drop,
            'mask_nouns_adjs_verb': self.encode_mask_nouns_adjs_verb,
            'mask_nouns_adjs_verb_replace': self.encode_mask_nouns_adjs_verb_replace,
            'mask_nouns_adjs_verb_replace_v2': self.encode_mask_nouns_adjs_verb_replace_2,
            'drop_nouns_adjs_verb_replace_v2': self.encode_drop_nouns_adjs_verb_replace_2,
            'mask_drop_nouns_adjs_verb_replace_v2': self.encode_mask_drop_nouns_adjs_verb_replace_2,
            'mask_drop_nouns_adjs_verb': self.encode_mask_drop_nouns_adjs_verb,
            'mask_nouns_adjs_verb_ablation_replace': self.encode_mask_nouns_adjs_verb_ablation_replace,
            'drop_nouns_adjs_verb_ablation_replace': self.encode_drop_nouns_adjs_verb_ablation_replace,
            'replace_nouns_adjs_verb_ablation_replace': self.encode_replace_nouns_adjs_verb_ablation_replace,
            'mask_replace_nouns_adjs_verb_ablation': self.encode_mask_replace_nouns_adjs_verb_ablation,
            'drop_replace_nouns_adjs_verb_ablation': self.encode_drop_replace_nouns_adjs_verb_ablation,
            'drop_mask_nouns_adjs_verb_ablation': self.encode_drop_mask_nouns_adjs_verb_ablation,
            'mask_drop_nouns_adjs_verb_replace_v3': self.encode_mask_drop_nouns_adjs_verb_replace_3,
            'mask_drop_nouns_adjs_verb_replace_v4': self.encode_mask_drop_nouns_adjs_verb_replace_4,
            'mask_drop_nouns_adjs_verb_replace_v5': self.encode_mask_drop_nouns_adjs_verb_replace_5,
            'mask_drop_nouns_replace_v5': self.encode_mask_drop_nouns_replace_5,
            'mask_drop_nouns_adjs_replace_replace_v5': self.encode_mask_drop_nouns_adjs_replace_5,
            'mask_nouns_adjs_verb_ablation_replace_v5': self.encode_mask_nouns_adjs_verb_ablation_replace_5,
            'drop_mask_nouns_adjs_verb_ablation_v5': self.encode_drop_mask_nouns_adjs_verb_ablation_5,
            'mask_drop_replace_v6': self.encode_mask_drop_replace_6,
            'mask_drop_nouns_replace_v6': self.encode_mask_drop_nouns_replace_6,
            'mask_drop_nouns_adjs_replace_replace_v6': self.encode_mask_drop_nouns_adjs_replace_6,
            'mask_nouns_adjs_verb_ablation_replace_v6': self.encode_mask_nouns_adjs_verb_ablation_replace_6,
            'drop_mask_nouns_adjs_verb_ablation_v6': self.encode_drop_mask_nouns_adjs_verb_ablation_6,
            'encode_mask_drop_replace_v6_random': self.encode_mask_drop_replace_6_random,
            # 'mask_drop_replace_njv_v7': self.encode_mask_drop_replace_njv_7,
            # 'mask_drop_replace_v7': self.encode_mask_drop_replace_7,
        }[encode_type]
        
        
        self.mask_prob_per_sen = mask_prob_per_sen
        self.mask_prob = mask_prob
        self.color_words_token = set()
        for token in self.color_words:
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpes = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            for bpe in bpes:
                self.color_words_token.add(bpe)
        self.color_words_token_with_index = {k: v + 1 for v, k in enumerate(self.color_words_token)}
        
        self.nouns_tokens = init_list("tools/list/cuhk_pedes/nouns.txt")
        self.nouns_tokens_with_index = {int(k): v + 1 for v, k in enumerate(self.nouns_tokens)}
        
        self.nouns_adjs_verb = init_list("tools/list/cuhk_pedes/nouns_adjs_verb.txt")
        self.nouns_adjs_verb_tokens_with_index = {int(k): v + 1 for v, k in enumerate(self.nouns_adjs_verb)}
        self.adjs_tokens = init_list("tools/list/cuhk_pedes/adjs.txt")
        self.verbs_tokens = init_list("tools/list/cuhk_pedes/verbs.txt")
        
        self.noun_words = init_list("tools/list/cuhk_pedes/noun_words.txt")
        self.verb_words = init_list("tools/list/cuhk_pedes/verb_words.txt")
        self.adj_words = init_list("tools/list/cuhk_pedes/adj_words.txt")
        
        self.drop_prob_per_sen = drop_prob_per_sen
        
        self.change_sen_prob = change_sen_prob
        # self.nlp = spacy.load("en_core_web_lg")
        
    # def most_similar(self, word, topn):
    #     words = []
    #     target = self.nlp.vocab.strings[word]
    #     if target in self.nlp.vocab.vectors:
    #         synonyms = self.nlp.vocab.vectors.most_similar(np.asarray([self.nlp.vocab.vectors[target]]), n=topn)
    #         words = [self.nlp.vocab.strings[w].lower() for w in synonyms[0][0] if self.nlp.vocab.strings[w].lower() != word.lower()]
    #     return words if words else word
        

    def cal_mask_pro(self, color):
        if color in self.color_words_high:
            return self.drop_high
        elif color in self.color_words_middle:
            return self.drop_middle
        else:
            return self.drop_low

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def remove_color_simple(self, text):
        drop = False
        result = []
        if self.on_sentence:
            r = torch.rand(1)
        for token in text:
            if token in self.color_words:
                if not self.on_sentence:
                    r = torch.rand(1)
                drop_pro = self.cal_mask_pro(token)
                if r < drop_pro:
                    drop = True
                    continue
            result.append(token)
        return result, drop

    def remove_color_with_link(self, text):
        drop = False
        result = []
        before_color = False
        before_removed = False
        if self.on_sentence:
            r = torch.rand(1)
        for token in text:
            if token in self.link:
                if before_removed: # [color](removed) {and}(remove)
                    continue
                # Do not change before_color
            elif token in self.color_words:
                if before_removed:  # [color](removed) and {color}(remove) // [color](removed) {color}(remove)
                    before_color = True
                    continue
                if not before_color:
                    if not self.on_sentence:
                        r = torch.rand(1)
                    drop_pro = self.cal_mask_pro(token)
                    if r < drop_pro:
                        before_removed = True
                        before_color = True
                        drop = True
                        continue
                # not before_removed and before_color # [color](not removed) {color}(skip) // [color](not removed) and {color}(skip)
                before_color = True
            else:
                before_color = False
            result.append(token)
            before_removed = False
        return result, drop

    def remove_color_one_per_sen(self, text):
        drop = False
        result = []
        before_color = False
        before_removed = False
        if self.on_sentence:
            r = torch.rand(1)
        for index, token in enumerate(text):
            if token in self.link:
                if before_removed: # [color](removed) {and}(remove)
                    continue
                # Do not change before_color
            elif token in self.color_words:
                if before_removed:  # [color](removed) and {color}(remove) // [color](removed) {color}(remove)
                    before_color = True
                    continue
                if not before_color:
                    if not self.on_sentence:
                        r = torch.rand(1)
                    drop_pro = self.cal_mask_pro(token)
                    if r < drop_pro:
                        before_removed = True
                        before_color = True
                        drop = True
                        continue
                # not before_removed and before_color # [color](not removed) {color}(skip) // [color](not removed) and {color}(skip)
                before_color = True
            else:
                before_color = False
            if before_removed:
                result.extend(text[index:])
                break
            result.append(token)
            before_removed = False
        return result, drop

    def remove_color_one_sen_prob(self, text):
        drop = False
        result = text
        before_color = False
        before_removed = False
        color_index = {'start': [], 'end': []}
        for index, token in enumerate(text):
            if token in self.link:
                if before_removed: # [color](removed) {and}(remove)
                    continue
                # Do not change before_color
            elif token in self.color_words:
                if before_removed:  # [color](removed) and {color}(remove) // [color](removed) {color}(remove)
                    before_color = True
                    continue
                if not before_color:
                    color_index['start'].append(index)
                    before_removed = True
                    before_color = True
                    continue
                # not before_removed and before_color # [color](not removed) {color}(skip) // [color](not removed) and {color}(skip)
                before_color = True
            else:
                before_color = False
            if before_removed:
                color_index['end'].append(index)
            before_removed = False
        if len(color_index['start']) != 0:
            r = torch.rand(1)
            if r < self.sen_drop_prob:
                drop = True
                index = random.randrange(len(color_index['start']))
                if len(color_index['start']) == len(color_index['end']):
                    result = text[:color_index['start'][index]] + text[color_index['end'][index]:]
                else:
                    result = text[:color_index['start'][index]]
        return result, drop
    
    def remove_color_one_sen_prob_label(self, text):
        drop = False
        result = text
        before_color = False
        before_removed = False
        color_index = {'start': [], 'end': []}
        for index, token in enumerate(text):
            if token in self.link:
                if before_removed: # [color](removed) {and}(remove)
                    continue
                # Do not change before_color
            elif token in self.color_words:
                if before_removed:  # [color](removed) and {color}(remove) // [color](removed) {color}(remove)
                    before_color = True
                    continue
                if not before_color:
                    color_index['start'].append(index)
                    before_removed = True
                    before_color = True
                    continue
                # not before_removed and before_color # [color](not removed) {color}(skip) // [color](not removed) and {color}(skip)
                before_color = True
            else:
                before_color = False
            if before_removed:
                color_index['end'].append(index)
            before_removed = False
        if len(color_index['start']) != 0:
            r = torch.rand(1)
            if r < self.sen_drop_prob:
                drop = True
                index = random.randrange(len(color_index['start']))
                if len(color_index['start']) == len(color_index['end']):
                    result = text[:color_index['start'][index]] + text[color_index['end'][index]:]
                else:
                    result = text[:color_index['start'][index]]
        id =  self.color_words_with_index[text[color_index['start'][index]]] if drop else -1
        return result, id
    
    def remove_color_one_sen_prob_label_simple(self, text):
        drop = False
        result = text
        color_index = []
        for index, token in enumerate(text):
            if token in self.color_words:
                color_index.append(index)
        if len(color_index) != 0:
            r = torch.rand(1)
            if r < self.sen_drop_prob:
                drop = True
                index = random.choice(color_index)
                result = text[:index] + text[index + 1:]
        id =  self.color_words_with_index[text[index]] if drop else -1
        return result, id
    
    def remove_color_multi_sen(self, text):
        drop = False
        result = text
        before_color = False
        before_removed = False
        color_index = {'start': [], 'end': []}
        for index, token in enumerate(text):
            if token in self.link:
                if before_removed: # [color](removed) {and}(remove)
                    continue
                # Do not change before_color
            elif token in self.color_words:
                if before_removed:  # [color](removed) and {color}(remove) // [color](removed) {color}(remove)
                    before_color = True
                    continue
                if not before_color:
                    color_index['start'].append(index)
                    before_removed = True
                    before_color = True
                    continue
                # not before_removed and before_color # [color](not removed) {color}(skip) // [color](not removed) and {color}(skip)
                before_color = True
            else:
                before_color = False
            if before_removed:
                color_index['end'].append(index)
            # result.append(token)
            before_removed = False
        if len(color_index['start']) != 0:
            r = torch.rand(1)
            if r < self.sen_drop_prob:
                drop = True
                limit = min(len(color_index['start']), self.per_sen_max)
                remove_num = random.randrange(1, limit + 1)
                selected_index = sorted(random.sample(range(0, len(color_index['start'])), remove_num))
                for index in range(remove_num):
                    if index == 0:
                        result += text[:color_index['start'][selected_index[index]]]
                    elif (index + 1) < remove_num:
                        result += (text[color_index['end'][selected_index[index - 1]]:color_index['start'][selected_index[index]]]
                        + text[color_index['end'][selected_index[index]]:color_index['start'][selected_index[index + 1]]])
                    else:
                        if len(color_index['start']) == len(color_index['end']):
                            result += text[color_index['end'][selected_index[index]]:]
        return result, drop

    def remove_color_multi_sen_simple(self, text):
        drop = False
        result = text
        color_index = []
        for index, token in enumerate(text):
            if token in self.color_words:
                color_index.append(index)
        if len(color_index) != 0:
            r = torch.rand(1)
            if r < self.sen_drop_prob:
                drop = True
                limit = min(len(color_index), self.per_sen_max)
                remove_num = random.randrange(1, limit + 1)
                selected_index = sorted(random.sample(color_index, remove_num))
                for index in range(remove_num):
                    if index == 0:
                        result += text[:selected_index[index]]
                    elif (index + 1) < remove_num:
                        result += text[selected_index[index - 1] + 1:selected_index[index]]
                    else:
                        result += (text[selected_index[index - 1] + 1:selected_index[index]]
                         + text[selected_index[index] + 1:])
        return result, drop

    def random_remove_fix_num(self, text):
        r = torch.rand(1)
        drop = False
        if r < self.random_remove_set[0]:
            drop = True
            idx = random.sample(range(len(text)), self.random_remove_set[1])
            del_list_inplace(text, idx)
        return text, drop

    def random_remove_max_num(self, text):
        r = torch.rand(1)
        drop = False
        if r < self.random_remove_set[0]:
            drop = True
            drop_num = random.randrange(1, self.random_remove_set[1] + 1)
            idx = random.sample(range(len(text)), drop_num)
            del_list_inplace(text, idx)
        return text, drop
    
    def random_remove_BERT(self, tokens):
        result = []
        for token in tokens:
            prob = torch.rand(1)
            if prob < self.mod_percent_per_sen:
                prob /= self.mod_percent_per_sen
                # 80% randomly change token to mask token
                if prob < self.BERT_drop_prob:
                    continue
                
            result.append(token)
        return result
    
    def random_remove_BERT_color(self, text):
        result = []
        drop = False
        before_color = False
        before_removed = False
        for token in text:
            if before_removed and ((token in self.link) or (before_color and token in self.color_words)):
                continue
            prob = torch.rand(1)
            if prob < self.mod_percent_per_sen:
                prob /= self.mod_percent_per_sen
                # 80% randomly change token to mask token
                if prob < self.BERT_drop_prob:
                    if token in self.color_words:
                        before_color = True if token in self.color_words else False
                    before_removed = True
                    continue
            result.append(token)
            before_color = True if token in self.color_words else False
            before_removed = False
        return result, drop
    
    def random_remove_BERT_color_without_link(self, text):
        result = []
        drop = False
        before_color = False
        before_removed = False
        for token in text:
            if before_removed and before_color and (token in self.color_words):
                continue
            prob = torch.rand(1)
            if prob < self.mod_percent_per_sen:
                prob /= self.mod_percent_per_sen
                # 80% randomly change token to mask token
                if prob < self.BERT_drop_prob:
                    if token in self.color_words:
                        before_color = True if token in self.color_words else False
                    before_removed = True
                    continue
            result.append(token)
            before_color = True if token in self.color_words else False
            before_removed = False
        return result, drop
    
    def random_remove_BERT_replace(self, tokens):
        result = []
        for token in tokens:
            prob = torch.rand(1)
            if prob < self.mod_percent_per_sen:
                prob /= self.mod_percent_per_sen

                # 80% randomly change token to mask token
                if prob < self.BERT_drop_prob:
                    continue
                
                # 10% randomly change token to random token
                elif prob < self.BERT_drop_prob + self.BERT_change_prob:
                    result.append(random.randrange(len(self.encoder)))
                    continue
                
            result.append(token)
        return result

    def random_mask_BERT_color(self, text):
        mask_label = []
        for i, token in enumerate(text):
            if token in self.color_words_token:
                prob = torch.rand(1)
                if prob < self.color_mask_percent_per_sen:
                    prob /= self.color_mask_percent_per_sen
                    if prob < self.mask_prob:
                        mask_label.append(self.color_words_token_with_index[token])
                        text[i] = self.encoder['<|maskoftext|>']
                        continue
            mask_label.append(0)
        return text, mask_label

    def random_mask_color(self, text):
        mask_label = []
        for i, token in enumerate(text):
            if token in self.color_words_token:
                prob = torch.rand(1)
                if prob < self.mask_prob_per_sen:
                    mask_label.append(self.color_words_token_with_index[token])
                    text[i] = self.encoder['<|maskoftext|>']
                    continue
            mask_label.append(0)
        return text, mask_label                   
                        
    def encode_default(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        drop = False
        if self.remove_color:
            text, drop = self.remove_f(text)
        for token in text:
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        if self.BERT_remove:
            bpe_tokens = self.remove_f(bpe_tokens)
            drop = True
        if self.mask_color:
            bpe_tokens, drop = self.mask_color_f(bpe_tokens)
        return bpe_tokens, drop
    
    def encode_mask_nouns(self, text):
        bpe_tokens = []
        mask_label = []
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if 'NN' in tag:
                prob = torch.rand(1)
                if prob < self.mask_prob_per_sen:
                    mask_label.extend(self.nouns_tokens_with_index[k] for k in bpe_token)
                    bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                    continue
            bpe_tokens.extend(bpe_token)
            mask_label.extend([0] * len(bpe_token))
        return bpe_tokens, mask_label
    
    def encode_nouns_adjs_drop(self, text):
        bpe_tokens = []
        mask_label = []
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mask_prob_per_sen:
                    continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    def encode_nouns_adjs_verb_drop(self, text):
        bpe_tokens = []
        mask_label = []
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mask_prob_per_sen:
                    continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    def encode_mask_nouns_adjs_verb(self, text):
        bpe_tokens = []
        mask_label = []
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mask_prob_per_sen:
                    mask_label.extend(self.nouns_adjs_verb_tokens_with_index[k] for k in bpe_token)
                    bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                    continue
            bpe_tokens.extend(bpe_token)
            mask_label.extend([0] * len(bpe_token))
        return bpe_tokens, mask_label
    
    def encode_drop_nouns_adjs_verb(self, text):
        bpe_tokens = []
        mask_label = []
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mask_prob_per_sen:
                    continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    def encode_mask_nouns_adjs_verb_replace(self, text):
        bpe_tokens = []
        mask_label = []
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mod_percent_per_sen:
                    prob /= self.mod_percent_per_sen
                    if prob < self.mask_prob_per_sen:
                        mask_label.extend(self.nouns_adjs_verb_tokens_with_index[k] for k in bpe_token)
                        bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                    else:
                        if tag in 'NN':
                            bpe_tokens.append(int(random.choice(self.nouns_tokens)))
                        elif tag in 'JJ':
                            bpe_tokens.append(int(random.choice(self.adjs_tokens)))
                        else:
                            bpe_tokens.append(int(random.choice(self.verbs_tokens)))
                        mask_label.extend([0])
                    continue
            bpe_tokens.extend(bpe_token)
            mask_label.extend([0] * len(bpe_token))
        return bpe_tokens, mask_label
    
    def encode_mask_nouns_adjs_verb_replace_2(self, text):
        bpe_tokens = []
        mask_label = []
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mod_percent_per_sen:
                    prob /= self.mod_percent_per_sen
                    if prob < self.mask_prob_per_sen:
                        mask_label.extend(self.nouns_adjs_verb_tokens_with_index[k] for k in bpe_token)
                        bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                    else:
                        if tag in 'NN':
                            token = random.choice(self.noun_words)
                        elif tag in 'JJ':
                            token = random.choice(self.adj_words)
                        else:
                            token = random.choice(self.verb_words)
                        bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                        bpe_tokens.extend(bpe_token)
                        mask_label.extend([0] * len(bpe_token))
                    continue
            bpe_tokens.extend(bpe_token)
            mask_label.extend([0] * len(bpe_token))
        return bpe_tokens, mask_label
    def encode_drop_nouns_adjs_verb_replace_2(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mod_percent_per_sen:
                    prob /= self.mod_percent_per_sen
                    if prob > self.mask_prob_per_sen:
                        if tag in 'NN':
                            token = random.choice(self.noun_words)
                        elif tag in 'JJ':
                            token = random.choice(self.adj_words)
                        else:
                            token = random.choice(self.verb_words)
                        bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                        bpe_tokens.extend(bpe_token)
                    continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    
    ########################################
    ### ------- ablation_replace ------- ###
    ########################################
    def encode_mask_nouns_adjs_verb_ablation_replace(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mod_percent_per_sen:
                    prob /= self.mod_percent_per_sen
                    if prob < self.mask_prob_per_sen:
                        # mask_label.extend(self.nouns_adjs_verb_tokens_with_index[k] for k in bpe_token)
                        bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                    elif prob > self.mask_prob_per_sen + self.drop_prob_per_sen:
                        if tag in 'NN':
                            token = random.choice(self.noun_words)
                        elif tag in 'JJ':
                            token = random.choice(self.adj_words)
                        else:
                            token = random.choice(self.verb_words)
                        # bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                        bpe_tokens.extend(bpe_token)
                    else:
                        bpe_token.extend(bpe_token)
                    continue
            bpe_tokens.extend(bpe_token)
            # mask_label.extend([0] * len(bpe_token))
        return bpe_tokens, mask_label
    def encode_mask_replace_nouns_adjs_verb_ablation(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mod_percent_per_sen:
                    prob /= self.mod_percent_per_sen
                    if prob < self.mask_prob_per_sen:
                        # mask_label.extend(self.nouns_adjs_verb_tokens_with_index[k] for k in bpe_token)
                        bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                    elif prob > self.mask_prob_per_sen + self.drop_prob_per_sen:
                        if tag in 'NN':
                            token = random.choice(self.noun_words)
                        elif tag in 'JJ':
                            token = random.choice(self.adj_words)
                        else:
                            token = random.choice(self.verb_words)
                        bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                        bpe_tokens.extend(bpe_token)
                    else:
                        bpe_tokens.extend(bpe_token)
                    continue
            bpe_tokens.extend(bpe_token)
            # mask_label.extend([0] * len(bpe_token))
        return bpe_tokens, mask_label
    
    def encode_drop_nouns_adjs_verb_ablation_replace(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mod_percent_per_sen:
                    prob /= self.mod_percent_per_sen
                    if prob < self.mask_prob_per_sen:
                        bpe_tokens.extend(bpe_token)
                    elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen:
                        continue
                    elif prob > self.mask_prob_per_sen + self.drop_prob_per_sen:
                        if tag in 'NN':
                            token = random.choice(self.noun_words)
                        elif tag in 'JJ':
                            token = random.choice(self.adj_words)
                        else:
                            token = random.choice(self.verb_words)
                        # bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                        bpe_tokens.extend(bpe_token)
                    else:
                        bpe_tokens.extend(bpe_token)
                    continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    def encode_drop_replace_nouns_adjs_verb_ablation(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mod_percent_per_sen:
                    prob /= self.mod_percent_per_sen
                    if prob < self.mask_prob_per_sen:
                        bpe_tokens.extend(bpe_token)
                    elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen:
                        continue
                    else:
                        if tag in 'NN':
                            token = random.choice(self.noun_words)
                        elif tag in 'JJ':
                            token = random.choice(self.adj_words)
                        else:
                            token = random.choice(self.verb_words)
                        bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                        bpe_tokens.extend(bpe_token)
                    continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    def encode_replace_nouns_adjs_verb_ablation_replace(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mod_percent_per_sen:
                    prob /= self.mod_percent_per_sen
                    if prob > self.mask_prob_per_sen + self.drop_prob_per_sen:
                        if tag in 'NN':
                            token = random.choice(self.noun_words)
                        elif tag in 'JJ':
                            token = random.choice(self.adj_words)
                        else:
                            token = random.choice(self.verb_words)
                        bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                        bpe_tokens.extend(bpe_token)
                    else:
                        bpe_tokens.extend(bpe_token)
                    continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    def encode_drop_mask_nouns_adjs_verb_ablation(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mod_percent_per_sen:
                    prob /= self.mod_percent_per_sen
                    if prob < self.mask_prob_per_sen:
                        bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                    elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen: # To be clear
                        continue
                    else:
                        if tag in 'NN':
                            token = random.choice(self.noun_words)
                        elif tag in 'JJ':
                            token = random.choice(self.adj_words)
                        else:
                            token = random.choice(self.verb_words)
                        # bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                        bpe_tokens.extend(bpe_token)
                    continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    
    # ORI
    def encode_mask_drop_nouns_adjs_verb_replace_2(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mod_percent_per_sen:
                    prob /= self.mod_percent_per_sen
                    if prob < self.mask_prob_per_sen:
                        bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                    elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen: # To be clear
                        continue
                    else:
                        if tag in 'NN':
                            token = random.choice(self.noun_words)
                        elif tag in 'JJ':
                            token = random.choice(self.adj_words)
                        else:
                            token = random.choice(self.verb_words)
                        bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                        bpe_tokens.extend(bpe_token)
                    continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    def encode_mask_drop_nouns_adjs_verb_replace_5(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mod_percent_per_sen:
                    prob /= self.mod_percent_per_sen
                    if prob < self.mask_prob_per_sen:
                        bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                    elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen: # To be clear
                        continue
                    else:
                        if tag in 'NN':
                            token = synonym_antonym_extractor(phrase=token, pos=wn.NOUN)
                        elif tag in 'JJ':
                            token = synonym_antonym_extractor(phrase=token, pos=wn.ADJ)
                        else:
                            token = synonym_antonym_extractor(phrase=token, pos=wn.VERB)
                        if '_' in token:
                            tokens = ' '.join(token.split('_'));
                            tokens = whitespace_clean(basic_clean(tokens)).lower()
                            tokens = re.findall(self.pat, tokens)
                            for token in tokens:
                                token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
                                bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                                bpe_tokens.extend(bpe_token)
                        else:
                            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                            bpe_tokens.extend(bpe_token)
                    continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    def encode_mask_drop_nouns_replace_5(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN']]):
                prob = torch.rand(1)
                if prob < self.mod_percent_per_sen:
                    prob /= self.mod_percent_per_sen
                    if prob < self.mask_prob_per_sen:
                        bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                    elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen: # To be clear
                        continue
                    else:
                        token = synonym_antonym_extractor(phrase=token, pos=wn.NOUN)
                        if '_' in token:
                            tokens = ' '.join(token.split('_'));
                            tokens = whitespace_clean(basic_clean(tokens)).lower()
                            tokens = re.findall(self.pat, tokens)
                            for token in tokens:
                                token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
                                bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                                bpe_tokens.extend(bpe_token)
                        else:
                            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                            bpe_tokens.extend(bpe_token)
                    continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    def encode_mask_drop_nouns_adjs_replace_5(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ']]):
                prob = torch.rand(1)
                if prob < self.mod_percent_per_sen:
                    prob /= self.mod_percent_per_sen
                    if prob < self.mask_prob_per_sen:
                        bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                    elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen: # To be clear
                        continue
                    else:
                        if tag in 'NN':
                            token = synonym_antonym_extractor(phrase=token, pos=wn.NOUN)
                        elif tag in 'JJ':
                            token = synonym_antonym_extractor(phrase=token, pos=wn.ADJ)
                        if '_' in token:
                            tokens = ' '.join(token.split('_'));
                            tokens = whitespace_clean(basic_clean(tokens)).lower()
                            tokens = re.findall(self.pat, tokens)
                            for token in tokens:
                                token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
                                bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                                bpe_tokens.extend(bpe_token)
                        else:
                            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                            bpe_tokens.extend(bpe_token)
                    continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    def encode_mask_nouns_adjs_verb_ablation_replace_5(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mod_percent_per_sen:
                    prob /= self.mod_percent_per_sen
                    if prob < self.mask_prob_per_sen:
                        # mask_label.extend(self.nouns_adjs_verb_tokens_with_index[k] for k in bpe_token)
                        bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                    elif prob > self.mask_prob_per_sen + self.drop_prob_per_sen:
                        if tag in 'NN':
                            token = synonym_antonym_extractor(phrase=token, pos=wn.NOUN)
                        elif tag in 'JJ':
                            token = synonym_antonym_extractor(phrase=token, pos=wn.ADJ)
                        else:
                            token = synonym_antonym_extractor(phrase=token, pos=wn.VERB)
                        # if '_' in token:
                        #     tokens = ' '.join(token.split('_'));
                        #     tokens = whitespace_clean(basic_clean(tokens)).lower()
                        #     tokens = re.findall(self.pat, tokens)
                        #     for token in tokens:
                        #         token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
                        #         bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                        #         bpe_tokens.extend(bpe_token)
                        # else:
                        #     bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                        bpe_tokens.extend(bpe_token)
                    else:
                        bpe_tokens.extend(bpe_token)
                    continue
            bpe_tokens.extend(bpe_token)
            # mask_label.extend([0] * len(bpe_token))
        return bpe_tokens, mask_label
    
    def encode_drop_mask_nouns_adjs_verb_ablation_5(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mod_percent_per_sen:
                    prob /= self.mod_percent_per_sen
                    if prob < self.mask_prob_per_sen:
                        bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                    elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen: # To be clear
                        continue
                    else:
                        if tag in 'NN':
                            token = synonym_antonym_extractor(phrase=token, pos=wn.NOUN)
                        elif tag in 'JJ':
                            token = synonym_antonym_extractor(phrase=token, pos=wn.ADJ)
                        else:
                            token = synonym_antonym_extractor(phrase=token, pos=wn.VERB)
                        # if '_' in token:
                        #     tokens = ' '.join(token.split('_'));
                        #     tokens = whitespace_clean(basic_clean(tokens)).lower()
                        #     tokens = re.findall(self.pat, tokens)
                        #     for token in tokens:
                        #         token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
                        #         bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                        #         bpe_tokens.extend(bpe_token)
                        # else:
                        #     bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                        bpe_tokens.extend(bpe_token)
                    continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    
    def encode_mask_drop_replace_6(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            prob = torch.rand(1)
            if prob < self.mod_percent_per_sen:
                prob /= self.mod_percent_per_sen
                if prob < self.mask_prob_per_sen:
                    bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen: # To be clear
                    continue
                else:
                    if tag in 'NN':
                        token = synonym_antonym_extractor(phrase=token, pos=wn.NOUN)
                    elif tag in 'JJ':
                        token = synonym_antonym_extractor(phrase=token, pos=wn.ADJ)
                    elif tag in 'VB':
                        token = synonym_antonym_extractor(phrase=token, pos=wn.VERB)
                    if '_' in token:
                        tokens = ' '.join(token.split('_'));
                        tokens = whitespace_clean(basic_clean(tokens)).lower()
                        tokens = re.findall(self.pat, tokens)
                        for token in tokens:
                            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
                            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                            bpe_tokens.extend(bpe_token)
                    else:
                        bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                        bpe_tokens.extend(bpe_token)
                continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    
    def encode_mask_drop_replace_6_random(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            prob = torch.rand(1)
            if prob < self.mod_percent_per_sen:
                prob /= self.mod_percent_per_sen
                if prob < self.mask_prob_per_sen:
                    bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen: # To be clear
                    continue
                else:
                    bpe_tokens.extend([random.randrange(len(self.encoder) - 3)])
                continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    def encode_mask_drop_nouns_replace_6(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            prob = torch.rand(1)
            if prob < self.mod_percent_per_sen:
                prob /= self.mod_percent_per_sen
                if prob < self.mask_prob_per_sen:
                    bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen: # To be clear
                    continue
                else:
                    if tag in 'NN':
                        token = synonym_antonym_extractor(phrase=token, pos=wn.NOUN)
                    if '_' in token:
                        tokens = ' '.join(token.split('_'));
                        tokens = whitespace_clean(basic_clean(tokens)).lower()
                        tokens = re.findall(self.pat, tokens)
                        for token in tokens:
                            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
                            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                            bpe_tokens.extend(bpe_token)
                    else:
                        bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                        bpe_tokens.extend(bpe_token)
                continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    def encode_mask_drop_nouns_adjs_replace_6(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            prob = torch.rand(1)
            if prob < self.mod_percent_per_sen:
                prob /= self.mod_percent_per_sen
                if prob < self.mask_prob_per_sen:
                    bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen: # To be clear
                    continue
                else:
                    if tag in 'NN':
                        token = synonym_antonym_extractor(phrase=token, pos=wn.NOUN)
                    elif tag in 'JJ':
                        token = synonym_antonym_extractor(phrase=token, pos=wn.ADJ)
                    if '_' in token:
                        tokens = ' '.join(token.split('_'));
                        tokens = whitespace_clean(basic_clean(tokens)).lower()
                        tokens = re.findall(self.pat, tokens)
                        for token in tokens:
                            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
                            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                            bpe_tokens.extend(bpe_token)
                    else:
                        bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                        bpe_tokens.extend(bpe_token)
                continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    def encode_mask_nouns_adjs_verb_ablation_replace_6(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            prob = torch.rand(1)
            if prob < self.mod_percent_per_sen:
                prob /= self.mod_percent_per_sen
                if prob < self.mask_prob_per_sen:
                    # mask_label.extend(self.nouns_adjs_verb_tokens_with_index[k] for k in bpe_token)
                    bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                elif prob > self.mask_prob_per_sen + self.drop_prob_per_sen:
                    if tag in 'NN':
                        token = synonym_antonym_extractor(phrase=token, pos=wn.NOUN)
                    elif tag in 'JJ':
                        token = synonym_antonym_extractor(phrase=token, pos=wn.ADJ)
                    elif tag in 'VB':
                        token = synonym_antonym_extractor(phrase=token, pos=wn.VERB)
                    # if '_' in token:
                    #     tokens = ' '.join(token.split('_'));
                    #     tokens = whitespace_clean(basic_clean(tokens)).lower()
                    #     tokens = re.findall(self.pat, tokens)
                    #     for token in tokens:
                    #         token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
                    #         bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                    #         bpe_tokens.extend(bpe_token)
                    # else:
                    #     bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                    bpe_tokens.extend(bpe_token)
                else:
                    bpe_tokens.extend(bpe_token)
                continue
            bpe_tokens.extend(bpe_token)
            # mask_label.extend([0] * len(bpe_token))
        return bpe_tokens, mask_label
    
    def encode_drop_mask_nouns_adjs_verb_ablation_6(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            prob = torch.rand(1)
            if prob < self.mod_percent_per_sen:
                prob /= self.mod_percent_per_sen
                if prob < self.mask_prob_per_sen:
                    bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen: # To be clear
                    continue
                else:
                    if tag in 'NN':
                        token = synonym_antonym_extractor(phrase=token, pos=wn.NOUN)
                    elif tag in 'JJ':
                        token = synonym_antonym_extractor(phrase=token, pos=wn.ADJ)
                    elif tag in 'VB':
                        token = synonym_antonym_extractor(phrase=token, pos=wn.VERB)
                    # if '_' in token:
                    #     tokens = ' '.join(token.split('_'));
                    #     tokens = whitespace_clean(basic_clean(tokens)).lower()
                    #     tokens = re.findall(self.pat, tokens)
                    #     for token in tokens:
                    #         token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
                    #         bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                    #         bpe_tokens.extend(bpe_token)
                    # else:
                    #     bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                    bpe_tokens.extend(bpe_token)
                continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label

    
    
    
    
    
    
    # def encode_mask_drop_replace_njv_7(self, text):
    #     bpe_tokens = []
    #     mask_label = True
    #     text = whitespace_clean(basic_clean(text)).lower()
    #     text = re.findall(self.pat, text)
    #     doc = self.nlp(' '.join(text))
    #     for token_tag in doc:
    #         token, tag = token_tag.text, token_tag.tag_
    #         token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
    #         bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
    #         prob = torch.rand(1)
    #         if any([x in tag for x in ['NN', 'JJ', 'VB']]):
    #             if prob < self.mod_percent_per_sen:
    #                 prob /= self.mod_percent_per_sen
    #                 if prob < self.mask_prob_per_sen:
    #                     bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
    #                 elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen: # To be clear
    #                     continue
    #                 else:
    #                     # # if any([x in tag for x in ['NN', 'JJ', 'VB']]):
    #                     # tokens = random.choice(self.most_similar(token, 5))
    #                     # # else:
    #                     # #     tokens = token
    #                     # tokens = whitespace_clean(basic_clean(tokens)).lower()
    #                     # tokens = re.findall(self.pat, tokens)
    #                     # for token in tokens:
    #                     #     token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
    #                     #     bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
    #                     #     bpe_tokens.extend(bpe_token)
    #                     bpe_tokens.append(random.randint(0, len(self.encoder) - 4))
    #                 continue
    #         bpe_tokens.extend(bpe_token)
    #     return bpe_tokens, mask_label
    
    # def encode_mask_drop_replace_7(self, text):
    #     bpe_tokens = []
    #     mask_label = True
    #     text = whitespace_clean(basic_clean(text)).lower()
    #     text = re.findall(self.pat, text)
    #     # doc = self.nlp(' '.join(text))
    #     for token in text:
    #     # for token_tag in doc:
    #         # token, tag = token_tag.text, token_tag.tag_
    #         token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
    #         bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
    #         prob = torch.rand(1)
    #         if prob < self.mod_percent_per_sen:
    #             prob /= self.mod_percent_per_sen
    #             if prob < self.mask_prob_per_sen:
    #                 bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
    #             elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen: # To be clear
    #                 continue
    #             else:
    #                 # if any([x in tag for x in ['NN', 'JJ', 'VB']]):
    #                 #     tokens = random.choice(self.most_similar(token, 5))
    #                 # else:
    #                 #     tokens = token
    #                 # tokens = whitespace_clean(basic_clean(tokens)).lower()
    #                 # tokens = re.findall(self.pat, tokens)
    #                 # for token in tokens:
    #                 #     token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
    #                 #     bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
    #                 #     bpe_tokens.extend(bpe_token)
    #                 bpe_tokens.append(random.randint(0, len(self.encoder) - 4))
    #             continue
    #         bpe_tokens.extend(bpe_token)
    #     return bpe_tokens, mask_label
    
    # def encode_mask_drop_nouns_adjs_verb_replace_7_v(self, text):
    #     text = whitespace_clean(basic_clean(text)).lower()
    #     text = re.findall(self.pat, text)
    #     result_hint = []
    #     result = []
    #     doc = self.nlp(' '.join(text))
    #     for token_tag in doc:
    #         token, tag = token_tag.text, token_tag.tag_
    #         token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            
            
    #         if any([x in tag for x in ['NN', 'JJ', 'VB']]):
    #             prob = torch.rand(1)
    #             if prob < self.mod_percent_per_sen:
    #                 prob /= self.mod_percent_per_sen
    #                 if prob < self.mask_prob_per_sen:
    #                     result_hint.append(f'[{token} -> [mask]]')
    #                     result.append('[mask]')
    #                 elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen: # To be clear
    #                     result_hint.append(f'[{token} -> [drop]]')
    #                 else:
    #                     ori_token = token
    #                     # if tag in 'NN':
    #                     # if any([x in tag for x in ['NN', 'JJ', 'VB']]):
    #                     token = random.choice(self.most_similar(token, 5))
    #                     # elif tag in 'JJ':
    #                     #     token = synonym_antonym_extractor(phrase=token, pos=wn.ADJ)
    #                     # else:
    #                     #     token = synonym_antonym_extractor(phrase=token, pos=wn.VERB)
    #                     # if tag in 'NN':
    #                     #         token = random.choice(self.noun_words)
    #                     # elif tag in 'JJ':
    #                     #     token = random.choice(self.adj_words)
    #                     # else:
    #                     #     token = random.choice(self.verb_words)
    #                     result_hint.append(f'[{ori_token} -> [{token}]]')
    #                     result.append(token)
    #                 continue
    #         result_hint.append(token)
    #         result.append(token)
    #     return ' '.join(result_hint), ' '.join(result)
    def encode_mask_drop_nouns_adjs_verb_replace_5_virual(self, text):
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        result_hint = []
        result = []
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mod_percent_per_sen:
                    prob /= self.mod_percent_per_sen
                    if prob < self.mask_prob_per_sen:
                        result_hint.append(f'[{token} -> [mask]]')
                        result.append('[mask]')
                    elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen: # To be clear
                        result_hint.append(f'[{token} -> [drop]]')
                    else:
                        ori_token = token
                        # if tag in 'NN':
                        token = synonym_antonym_extractor(phrase=token, pos=tag)
                        # elif tag in 'JJ':
                        #     token = synonym_antonym_extractor(phrase=token, pos=wn.ADJ)
                        # else:
                        #     token = synonym_antonym_extractor(phrase=token, pos=wn.VERB)
                        # if tag in 'NN':
                        #         token = random.choice(self.noun_words)
                        # elif tag in 'JJ':
                        #     token = random.choice(self.adj_words)
                        # else:
                        #     token = random.choice(self.verb_words)
                        result_hint.append(f'[{ori_token} -> [{token}]]')
                        result.append(token)
                    continue
            result_hint.append(token)
            result.append(token)
        return ' '.join(result_hint), ' '.join(result)
    
    def encode_mask_drop_nouns_adjs_verb_replace_4(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mod_percent_per_sen:
                    prob /= self.mod_percent_per_sen
                    if prob < self.mask_prob_per_sen:
                        bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                    elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen: # To be clear
                        continue
                    else:
                        if tag in 'NN':
                            token = synonym_antonym_extractor(phrase=token, pos=wn.NOUN)
                        elif tag in 'JJ':
                            token = synonym_antonym_extractor(phrase=token, pos=wn.ADJ)
                        else:
                            token = synonym_antonym_extractor(phrase=token, pos=wn.VERB)
                        bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                        bpe_tokens.extend(bpe_token)
                    continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    def encode_mask_drop_nouns_adjs_verb_replace_3(self, text):
        bpe_tokens = []
        mask_label = True
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        sen_prob = torch.rand(1)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if sen_prob < self.change_sen_prob:
                if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                    prob = torch.rand(1)
                    if prob < self.mod_percent_per_sen:
                        prob /= self.mod_percent_per_sen
                        if prob < self.mask_prob_per_sen:
                            bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                        elif prob < self.mask_prob_per_sen + self.drop_prob_per_sen: # To be clear
                            continue
                        else:
                            if tag in 'NN':
                                token = random.choice(self.noun_words)
                            elif tag in 'JJ':
                                token = random.choice(self.adj_words)
                            else:
                                token = random.choice(self.verb_words)
                            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
                            bpe_tokens.extend(bpe_token)
                        continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    def encode_mask_drop_nouns_adjs_verb(self, text):
        bpe_tokens = []
        mask_label = []
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if any([x in tag for x in ['NN', 'JJ', 'VB']]):
                prob = torch.rand(1)
                if prob < self.mod_percent_per_sen:
                    prob /= self.mod_percent_per_sen
                    if prob < self.mask_prob_per_sen:
                        mask_label.extend(self.nouns_adjs_verb_tokens_with_index[k] for k in bpe_token)
                        bpe_tokens.extend([self.encoder['<|maskoftext|>']] * len(bpe_token))
                    continue
            bpe_tokens.extend(bpe_token)
            mask_label.extend([0] * len(bpe_token))
        return bpe_tokens, mask_label
    
    def encode_adjs_drop(self, text):
        bpe_tokens = []
        mask_label = []
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if 'JJ' in tag:
                prob = torch.rand(1)
                if prob < self.mask_prob_per_sen:
                    continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    def encode_nouns_drop(self, text):
        bpe_tokens = []
        mask_label = []
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        pos_tag = nltk.pos_tag(text)
        for token_tag in pos_tag:
            token, tag = token_tag
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            if 'NN' in tag:
                prob = torch.rand(1)
                if prob < self.mask_prob_per_sen:
                    continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label
    
    def encode_BERT_like_drop(self, text):
        bpe_tokens = []
        mask_label = []
        text = whitespace_clean(basic_clean(text)).lower()
        text = re.findall(self.pat, text)
        for token in text:
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            prob = torch.rand(1)
            if prob < self.mask_prob_per_sen:
                continue
            bpe_tokens.extend(bpe_token)
        return bpe_tokens, mask_label

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text