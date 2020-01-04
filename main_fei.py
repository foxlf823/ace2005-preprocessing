import os
import copy
import re
import json
from stanfordcorenlp import StanfordCoreNLP
import argparse
from tqdm import tqdm

verbose = False
pattern = re.compile(r'[-]')
pattern1 = r'([-])'

class Tree(object):
    def __init__(self, nodes):
        self.nodes = nodes
        self.match = True

        self.leaf_nodes = []
        for node in self.nodes:
            # if node.data != "":
            if len(node.children) == 0 and node.data != "":  # some leaf nodes don't correspond to tokens
                self.leaf_nodes.append(node)

    def get_span_for_leaf_node(self, sentence):
        if len(self.leaf_nodes) != len(sentence):
            self.match = False
            for idx, node in enumerate(self.leaf_nodes):
                node.span = [idx, idx]  # inclusive endpoints
        else:
            for idx, (node, text) in enumerate(zip(self.leaf_nodes, sentence)):
                if node.data != text:
                    self.match = False
                node.span = [idx, idx]  # inclusive endpoints


    def get_span_for_node(self, sentence):

        # assert (len(sentence) == len(self.leaf_nodes))
        # for idx, (node, text) in enumerate(zip(self.leaf_nodes, sentence)):
        #     assert (node.data == text)
        #     node.span = [idx, idx]  # inclusive endpoints

        sentence_start, sentence_end = Tree.get_span_for_node_(self.nodes, 0)
        # assert(sentence_start==0)
        # assert(sentence_end==len(sentence)-1)

        for node in self.nodes:
            if node.span[1] == -1: # update the span for Null constituent
                node.span[0] = -1

    def show_leaf_node(self):
        ret = [n.data for n in self.leaf_nodes]
        return ret


    @staticmethod
    def get_span_for_node_(nodes, node_idx):
        node = nodes[node_idx]
        if len(node.children) == 0:
            if node.data != "":
                return node.span[0], node.span[1]
            else:
                node.span = [999, -1]
                return 999, -1
        else:
            span_start = 999
            span_end = -1
            for child_idx in node.children:
                child_span_start, child_span_end = Tree.get_span_for_node_(nodes, child_idx)
                span_start = child_span_start if child_span_start < span_start else span_start
                span_end = child_span_end if child_span_end > span_end else span_end

            # assert(span_start<=span_end)
            node.span = [span_start, span_end]
            return span_start, span_end

    def print_tree(self):
        return Tree._print_node(self.nodes, 0, "", 0)

    @staticmethod
    def _print_node(nodes, node_idx, ret, level):
        node = nodes[node_idx]
        indent = '  '*level
        ret += '\n'+indent+'('+node.cat if node.data == '' else ' ('+node.cat+' '+node.data
        for child_idx in node.children:
            ret = Tree._print_node(nodes, child_idx, ret, level+1)
        ret += ')'
        return ret

    def to_json(self):
        nodes = []
        ret = {'match':self.match, 'nodes': nodes}
        for node in self.nodes:
            nodes.append(node.to_json())
        return ret

class Node(object):
    def __init__(self, parent, cat, data):
        self.parent = parent
        self.cat = cat
        self.data = data
        self.children = []

    def __str__(self):
        return self.cat+"|"+self.data+"|"+str(self.parent)

    def to_json(self):
        return [self.parent, self.children, self.cat, self.data, self.span]

def get_node(input, nodes, parent_idx, tokenized):
    if isinstance(input[-1], str):
        if tokenized:
            tokens = [tok for tok in re.split(pattern1, input[1])
                      if tok not in ["", " "]]
            for token in tokens:
                node = Node(parent_idx, input[0], token)
                nodes.append(node)
                nodes[parent_idx].children.append(len(nodes) - 1)
        else:
            node = Node(parent_idx, input[0], input[1])
            nodes.append(node)
            nodes[parent_idx].children.append(len(nodes) - 1)
    else:
        node = Node(parent_idx, input[0], "")
        nodes.append(node)
        if parent_idx >= 0:
            nodes[parent_idx].children.append(len(nodes) - 1)
        new_parent_idx = len(nodes) - 1
        for child in input[1:]:
            get_node(child, nodes, new_parent_idx, tokenized)

def readTree(text, ind, verbose=False):
    """The basic idea here is to represent the file contents as a long string
    and iterate through it character-by-character (the 'ind' variable
    points to the current character). Whenever we get to a new tree,
    we call the function again (recursively) to read it in."""
    # if verbose:
    #     print("Reading new subtree", text[ind:][:10])

    # consume any spaces before the tree
    while text[ind].isspace():
        ind += 1

    if text[ind] == "(":
        # if verbose:
        #     print("Found open paren")
        tree = []
        ind += 1

        # record the label after the paren
        label = ""
        while not text[ind].isspace() and text != "(":
            label += text[ind]
            ind += 1

        tree.append(label)
        # if verbose:
        #     print("Read in label:", label)

        # read in all subtrees until right paren
        subtree = True
        while subtree:
            # if this call finds only the right paren it'll return False
            subtree, ind = readTree(text, ind, verbose=verbose)
            if subtree:
                tree.append(subtree)

        # consume the right paren itself
        ind += 1
        assert(text[ind] == ")")
        ind += 1

        # if verbose:
        #     print("End of tree", tree)

        return tree, ind

    elif text[ind] == ")":
        # there is no subtree here; this is the end paren of the parent tree
        # which we should not consume
        ind -= 1
        return False, ind

    else:
        # the subtree is just a terminal (a word)
        word = ""
        while not text[ind].isspace() and text[ind] != ")":
            word += text[ind]
            ind += 1

        # if verbose:
        #     print("Read in word:", word)

        return word, ind


def get_data_paths(ace2005_path, data_list_path):
    test_files, dev_files, train_files = [], [], []
    # with open('./data_list.csv', mode='r') as csv_file:
    # with open('./data_list_test.csv', mode='r') as csv_file:
    with open(data_list_path, mode='r') as csv_file:
        rows = csv_file.readlines()
        for row in rows[1:]:
            items = row.replace('\n', '').split(',')
            data_type = items[0]
            name = items[1]

            path = os.path.join(ace2005_path, name)
            if data_type == 'test':
                test_files.append(path)
            elif data_type == 'dev':
                dev_files.append(path)
            elif data_type == 'train':
                train_files.append(path)
    return test_files, dev_files, train_files


def find_token_index(tokens, start_pos, end_pos, phrase):
    start_idx, end_idx = -1, -1
    for idx, token in enumerate(tokens):
        if token['start'] == start_pos:
            start_idx = idx
        if token['end'] == end_pos:
            end_idx = idx
        if start_idx != -1 and end_idx != -1:
            break

    # assert start_idx != -1, "start_idx: {}, start_pos: {}, phrase: {}, tokens: {}".format(start_idx, start_pos, phrase, tokens)
    # assert end_idx != -1, "end_idx: {}, end_pos: {}, phrase: {}, tokens: {}".format(end_idx, end_pos, phrase,
    #                                                                                           tokens)
    return start_idx, end_idx # inclusive endpoint


def verify_result(results):
    def remove_punctuation(s):
        for c in ['-LRB-', '-RRB-', '-LSB-', '-RSB-', '-LCB-', '-RCB-', '\xa0']:
            s = s.replace(c, '')
        s = re.sub(r'[^\w]', '', s)
        return s

    def check_diff(words, phrase):
        return remove_punctuation(phrase) not in remove_punctuation(words)

    for doc in results:
        data = doc['sentences']
        for item in data:
            words = item['words']
            for entity_mention in item['golden-entity-mentions']:
                if check_diff(''.join(words[entity_mention['start']:entity_mention['end']]), entity_mention['text'].replace(' ', '')):
                    print('============================')
                    print('[Warning] entity has invalid start/end')
                    print('Expected: ', entity_mention['text'])
                    print('Actual:', words[entity_mention['start']:entity_mention['end']])
                    print('start: {}, end: {}, words: {}'.format(entity_mention['start'], entity_mention['end'], words))


    print('Complete verification')


def my_split(s):
    text = []
    offset = []
    iter = re.finditer(pattern, s)
    start = 0
    for i in iter:
        if start != i.start():
            text.append(s[start: i.start()])
            offset.append(start)
        text.append(s[i.start(): i.end()])
        offset.append(i.start())
        start = i.end()
    if start != len(s):
        text.append(s[start: ])
        offset.append(start)
    return text, offset

from xml.etree import ElementTree
from bs4 import BeautifulSoup
import json
import re
import nltk

class Parser:
    def __init__(self, stats, path):
        self.path = path
        self.entity_mentions = []
        self.sgm_text = ''

        self.entity_mentions = self.parse_xml(path + '.apf.xml')
        self.sents_with_pos = self.parse_sgm(path + '.sgm')
        self.fix_wrong_position()

        stats['original_entity'] += len(self.entity_mentions)
        stats['original_sent'] += len(self.sents_with_pos)
        # remove entitiies other than ['FAC','GPE','LOC','ORG','PER','VEH','WEA']
        # remove reduplicative entity
        entities = []
        entity_span = set()
        for entity_mention in self.entity_mentions:
            if entity_mention['entity-type'] not in ['FAC','GPE','LOC','ORG','PER','VEH','WEA']:
                continue

            if (entity_mention['position'][0], entity_mention['position'][1]) in entity_span:
                # reduplicative entity, ignore
                continue
            else:
                entity_span.add((entity_mention['position'][0], entity_mention['position'][1]))
                entities.append(entity_mention)
        self.entity_mentions = entities
        stats['entity_type_redup_removed'] += len(self.entity_mentions)

    @staticmethod
    def clean_text(text):
        return text.replace('\n', ' ')

    def get_data(self):
        data = []
        for sent in self.sents_with_pos:
            item = dict()

            item['sentence'] = self.clean_text(sent['text'])
            item['position'] = sent['position']
            text_position = sent['position']

            for i, s in enumerate(item['sentence']):
                if s != ' ':
                    item['position'][0] += i
                    break

            item['sentence'] = item['sentence'].strip()

            item['golden-entity-mentions'] = []

            for entity_mention in self.entity_mentions:
                entity_position = entity_mention['position']
                if text_position[0] <= entity_position[0] and entity_position[1] <= text_position[1]:
                    item['golden-entity-mentions'].append({
                        'text': self.clean_text(entity_mention['text']),
                        'position': entity_position,
                        'entity-type': entity_mention['entity-type']
                    })

            data.append(item)
        return data

    def find_correct_offset(self, sgm_text, start_index, text):
        offset = 0
        for i in range(0, 70):
            for j in [-1, 1]:
                offset = i * j
                if sgm_text[start_index + offset:start_index + offset + len(text)] == text:
                    return offset

        if verbose:
            print('[Warning] fail to find offset! (start_index: {}, text: {}, path: {})'.format(start_index, text, self.path))
        raise RuntimeError

    def fix_wrong_position(self):
        for entity_mention in self.entity_mentions:
            offset = self.find_correct_offset(
                sgm_text=self.sgm_text,
                start_index=entity_mention['position'][0],
                text=entity_mention['text'])

            entity_mention['position'][0] += offset
            entity_mention['position'][1] += offset

    def parse_sgm(self, sgm_path):
        with open(sgm_path, 'r') as f:
            soup = BeautifulSoup(f.read(), features='html.parser')
            self.sgm_text = soup.text

            doc_type = soup.doc.doctype.text.strip()

            def remove_tags(selector):
                tags = soup.findAll(selector)
                for tag in tags:
                    tag.extract()

            if doc_type == 'WEB TEXT':
                remove_tags('poster')
                remove_tags('postdate')
                remove_tags('subject')
            elif doc_type in ['CONVERSATION', 'STORY']:
                remove_tags('speaker')

            sents = []
            converted_text = soup.text

            for sent in nltk.sent_tokenize(converted_text):
                sents.extend(sent.split('\n\n'))
            sents = list(filter(lambda x: len(x) > 5, sents))
            sents = sents[1:]
            sents_with_pos = []
            last_pos = 0
            for sent in sents:
                pos = self.sgm_text.find(sent, last_pos)
                last_pos = pos
                sents_with_pos.append({
                    'text': sent,
                    'position': [pos, pos + len(sent)] # exclusive endpoint
                })

            return sents_with_pos

    def parse_xml(self, xml_path):
        entity_mentions = []
        tree = ElementTree.parse(xml_path)
        root = tree.getroot()

        for child in root[0]:
            if child.tag == 'entity':
                entity_mentions.extend(self.parse_entity_tag(child))

        return entity_mentions

    @staticmethod
    def parse_entity_tag(node):
        entity_mentions = []

        for child in node:
            if child.tag != 'entity_mention':
                continue
            extent = child[0]
            charset = extent[0]

            entity_mention = dict()
            entity_mention['entity-id'] = child.attrib['ID']
            entity_mention['entity-type'] = node.attrib['TYPE']
            entity_mention['text'] = charset.text
            entity_mention['position'] = [int(charset.attrib['START']), int(charset.attrib['END'])] # inclusive endpoint

            entity_mentions.append(entity_mention)

        return entity_mentions

# from parser import Parser
def preprocessing(data_type, files):

    stats = dict(original_entity=0, original_sent=0, entity_type_redup_removed=0, entity=0, sentence=0, entity_common=0)
    stats_treebank = dict(sent_mismatches=0, sent_matches=0)
    fp = open('output/{}.json'.format(data_type), 'w')
    print('=' * 20)
    print('[preprocessing] type: ', data_type)

    for file in tqdm(files):
        parser = Parser(stats, path=file)

        doc = dict(name=file[file.rfind("/")+1:])
        # if doc['name'] == 'APW_ENG_20030419.0358':
        #     a = 1
        doc['sentences'] = []

        for item in parser.get_data():
            data = dict()

            data['golden-entity-mentions'] = []
            try:
                nlp_res_raw = nlp.annotate(item['sentence'], properties={'annotators': 'tokenize,ssplit,pos,parse'})
                nlp_res = json.loads(nlp_res_raw)
            except Exception as e:
                if verbose:
                    print('[Warning] StanfordCore Timeout: ', item['sentence'])
                # print('If you want to include all sentences, please refer to this issue: https://github.com/nlpcl-lab/ace2005-preprocessing/issues/1')
                continue

            if len(nlp_res['sentences']) >= 2:
                # TODO: issue where the sentence segmentation of NTLK and StandfordCoreNLP do not match
                # This error occurred so little that it was temporarily ignored (< 20 sentences).
                if verbose:
                    print('[Warning] sentence segmentation of NTLK and StandfordCoreNLP do not match: ', item['sentence'])
                continue

            # adjust tokenization
            original_tokens = nlp_res['sentences'][0]['tokens']
            tokens = []
            for x in original_tokens:
                # if x['word'].find('-') != -1:
                #     a = 1
                split_tokens, split_offsets  = my_split(x['word'])
                for st, so in zip(split_tokens, split_offsets):
                    token = {}
                    token['word'] = st
                    token['start'] = x['characterOffsetBegin'] + so
                    token['end'] = token['start'] + len(st)
                    tokens.append(token)

            data['words'] = tokens

            data['parse'] = nlp_res['sentences'][0]['parse']

            sent_start_pos = item['position'][0]

            for entity_mention in item['golden-entity-mentions']:
                position = entity_mention['position']
                start_idx, end_idx = find_token_index(
                    tokens=tokens,
                    start_pos=position[0] - sent_start_pos,
                    end_pos=position[1] - sent_start_pos + 1,
                    phrase=entity_mention['text'],
                )

                if start_idx == -1:
                    if verbose:
                        print("tokenize and annotation mismatch: ", "start_idx: {}, start_pos: {}, phrase: {}, tokens: {}".format(start_idx, position[0] - sent_start_pos, entity_mention['text'], tokens))
                    continue

                if end_idx == -1:
                    if verbose:
                        print("tokenize and annotation mismatch: ", "end_idx: {}, end_pos: {}, phrase: {}, tokens: {}".format(end_idx, position[1] - sent_start_pos + 1, entity_mention['text'], tokens))
                    continue

                entity_mention['start'] = start_idx
                entity_mention['end'] = end_idx

                data['golden-entity-mentions'].append(entity_mention)
                if entity_mention['end'] - entity_mention['start'] + 1 <= 15:
                    stats['entity_common'] += 1
                stats['entity'] += 1

            doc['sentences'].append(data)
            stats['sentence'] += 1

        # transfer into dygie json format
        dygie_doc = {}
        dygie_doc['doc_key'] = doc['name'] + '_' + data_type
        dygie_doc['sentences'] = []
        dygie_doc['ner'] = []
        dygie_doc['trees'] = []
        offset = 0
        for sentence in doc['sentences']:
            dygie_sentence = [w['word'] for w in sentence['words']]
            dygie_doc['sentences'].append(dygie_sentence)

            entity_of_this_sentence = []
            for entity in sentence['golden-entity-mentions']:
                dygie_entity = []
                dygie_entity.append(entity['start'] + offset)
                dygie_entity.append(entity['end'] + offset)  # inclusive endpoint
                dygie_entity.append(entity['entity-type'])
                entity_of_this_sentence.append(dygie_entity)
            dygie_doc['ner'].append(entity_of_this_sentence)
            offset += len(sentence['words'])

            tree, _ = readTree(sentence['parse'], 0)
            nodes = []
            get_node(tree, nodes, -1, True)
            tree = Tree(nodes)
            tree.get_span_for_leaf_node(dygie_sentence)
            if not tree.match:
                if verbose:
                    print("sent mismatch, doc: {}, original sentence: {}, tree sentence {}".format(doc['name'], sentence['words'], tree.show_leaf_node()))
                stats_treebank['sent_mismatches'] += 1
            else:
                stats_treebank['sent_matches'] += 1
            tree.get_span_for_node(dygie_sentence)
            dygie_doc['trees'].append(tree.to_json())

        fp.write(json.dumps(dygie_doc) + "\n")

    print(stats)
    print(stats_treebank)
    fp.close()
    return

def generate_data_list(input_dir, output_file):
    fp = open(output_file, 'w')
    fp.write("type,path\n")
    for fold in ['train', 'dev', 'test']:
        dir = os.path.join(input_dir, fold)
        for file_name in os.listdir(dir):
            if file_name.find('.apf.xml') != -1:
                short_name = file_name[:file_name.rfind('.apf.xml')]
                fp.write(fold+','+fold+'/'+short_name+"\n")
    fp.close()

def generate_data_list_1(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for fold in ['train', 'dev', 'test']:
        fp = open(os.path.join(output_dir, fold), 'w')
        dir = os.path.join(input_dir, fold)
        for file_name in os.listdir(dir):
            if file_name.find('.apf.xml') != -1:
                short_name = file_name[:file_name.rfind('.apf.xml')]
                fp.write(short_name+"\n")
        fp.close()

def generate_train_list_and_fixed(input_dir):
    # input_dir should only contain bc bn nw wl
    fp1 = open('./train_list', 'w')
    fp2 = open('./train_list_fixed', 'w')
    for dir in os.listdir(input_dir):
        if dir in ['bc', 'bn', 'nw', 'wl']:
            for file_name in os.listdir(os.path.join(input_dir, dir, 'timex2norm')):
                if file_name.find('.apf.xml') != -1:
                    short_name = file_name[:file_name.rfind('.apf.xml')]
                    fp1.write("result/"+short_name+".txt"+"\n")
                    fp2.write("fixed/"+short_name+".txt"+"\n")
    fp1.close()
    fp2.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help="Path of ACE2005 English data", default='./data/ace_2005_td_v7/data/English')
    args = parser.parse_args()

    # generate_data_list(args.data, './data_list_fei.csv')
    # generate_data_list_1(args.data, './split')
    # generate_train_list_and_fixed('/Users/feili/dataset/ACE2005-TrainingData-V5.0/English')
    # exit(1)

    test_files, dev_files, train_files = get_data_paths(args.data, './data_list_fei.csv')

    with StanfordCoreNLP('./stanford-corenlp-full-2018-10-05', memory='8g', timeout=60000) as nlp:
    # with StanfordCoreNLP('./stanford-corenlp-full-2015-04-20', memory='8g', timeout=60000) as nlp:
        # res = nlp.annotate('Donald John Trump is current president of the United States.', properties={'annotators': 'tokenize,ssplit,pos,lemma,parse'})
        # print(res)

        preprocessing('dev', dev_files)
        preprocessing('test', test_files)
        preprocessing('train', train_files)
