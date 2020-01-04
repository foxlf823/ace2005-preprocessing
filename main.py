import os
import copy
import re
from parser import Parser
import json
from stanfordcorenlp import StanfordCoreNLP
import argparse
from tqdm import tqdm

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

def get_node(input, nodes, parent_idx):
    if isinstance(input[-1], str):
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
            get_node(child, nodes, new_parent_idx)

def readTree(text, ind, verbose=False):
    """The basic idea here is to represent the file contents as a long string
    and iterate through it character-by-character (the 'ind' variable
    points to the current character). Whenever we get to a new tree,
    we call the function again (recursively) to read it in."""
    if verbose:
        print("Reading new subtree", text[ind:][:10])

    # consume any spaces before the tree
    while text[ind].isspace():
        ind += 1

    if text[ind] == "(":
        if verbose:
            print("Found open paren")
        tree = []
        ind += 1

        # record the label after the paren
        label = ""
        while not text[ind].isspace() and text != "(":
            label += text[ind]
            ind += 1

        tree.append(label)
        if verbose:
            print("Read in label:", label)

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

        if verbose:
            print("End of tree", tree)

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

        if verbose:
            print("Read in word:", word)

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
        if token['characterOffsetBegin'] <= start_pos:
            start_idx = idx

    assert start_idx != -1, "start_idx: {}, start_pos: {}, phrase: {}, tokens: {}".format(start_idx, start_pos, phrase, tokens)
    chars = ''

    def remove_punc(s):
        s = re.sub(r'[^\w]', '', s)
        return s

    for i in range(0, len(tokens) - start_idx):
        chars += remove_punc(tokens[start_idx + i]['originalText'])
        if remove_punc(phrase) in chars:
            end_idx = start_idx + i + 1
            break

    assert end_idx != -1, "end_idx: {}, end_pos: {}, phrase: {}, tokens: {}, chars:{}".format(end_idx, end_pos, phrase, tokens, chars)
    return start_idx, end_idx


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

            # for event_mention in item['golden-event-mentions']:
            #     trigger = event_mention['trigger']
            #     if check_diff(''.join(words[trigger['start']:trigger['end']]), trigger['text'].replace(' ', '')):
            #         print('============================')
            #         print('[Warning] trigger has invalid start/end')
            #         print('Expected: ', trigger['text'])
            #         print('Actual:', words[trigger['start']:trigger['end']])
            #         print('start: {}, end: {}, words: {}'.format(trigger['start'], trigger['end'], words))
            #     for argument in event_mention['arguments']:
            #         if check_diff(''.join(words[argument['start']:argument['end']]), argument['text'].replace(' ', '')):
            #             print('============================')
            #             print('[Warning] argument has invalid start/end')
            #             print('Expected: ', argument['text'])
            #             print('Actual:', words[argument['start']:argument['end']])
            #             print('start: {}, end: {}, words: {}'.format(argument['start'], argument['end'], words))

    print('Complete verification')


def preprocessing(data_type, files):
    result = []
    # event_count, entity_count, sent_count, argument_count = 0, 0, 0, 0

    print('=' * 20)
    print('[preprocessing] type: ', data_type)
    for file in tqdm(files):
        parser = Parser(path=file)

        # entity_count += len(parser.entity_mentions)
        # event_count += len(parser.event_mentions)
        # sent_count += len(parser.sents_with_pos)

        doc = dict(name=file[file.rfind("/")+1:])
        sentences = []
        doc['sentences'] = sentences

        for item in parser.get_data():
            data = dict()

            # data['sentence'] = item['sentence']
            data['golden-entity-mentions'] = []
            # data['golden-event-mentions'] = []

            try:
                nlp_res_raw = nlp.annotate(item['sentence'], properties={'annotators': 'tokenize,ssplit,pos,lemma,parse'})
                nlp_res = json.loads(nlp_res_raw)
            except Exception as e:
                print('[Warning] StanfordCore Exception: ', nlp_res_raw, 'This sentence will be ignored.')
                print('If you want to include all sentences, please refer to this issue: https://github.com/nlpcl-lab/ace2005-preprocessing/issues/1')
                continue

            tokens = nlp_res['sentences'][0]['tokens']

            if len(nlp_res['sentences']) >= 2:
                # TODO: issue where the sentence segmentation of NTLK and StandfordCoreNLP do not match
                # This error occurred so little that it was temporarily ignored (< 20 sentences).
                continue

            # data['stanford-colcc'] = []
            # for dep in nlp_res['sentences'][0]['enhancedPlusPlusDependencies']:
            #     data['stanford-colcc'].append('{}/dep={}/gov={}'.format(dep['dep'], dep['dependent'] - 1, dep['governor'] - 1))

            data['words'] = list(map(lambda x: x['word'], tokens))
            # data['pos-tags'] = list(map(lambda x: x['pos'], tokens))
            # data['lemma'] = list(map(lambda x: x['lemma'], tokens))
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

                entity_mention['start'] = start_idx
                entity_mention['end'] = end_idx

                del entity_mention['position']

                entity_mention['type'] = entity_mention['entity-type'].split(':')[0]
                if entity_mention['type'] not in ['FAC','GPE','LOC','ORG','PER','VEH','WEA']:
                    continue
                del entity_mention['entity-type']

                data['golden-entity-mentions'].append(entity_mention)

            # for event_mention in item['golden-event-mentions']:
            #     # same event mention can be shared
            #     event_mention = copy.deepcopy(event_mention)
            #     position = event_mention['trigger']['position']
            #     start_idx, end_idx = find_token_index(
            #         tokens=tokens,
            #         start_pos=position[0] - sent_start_pos,
            #         end_pos=position[1] - sent_start_pos + 1,
            #         phrase=event_mention['trigger']['text'],
            #     )
            #
            #     event_mention['trigger']['start'] = start_idx
            #     event_mention['trigger']['end'] = end_idx
            #     del event_mention['trigger']['position']
            #     del event_mention['position']
            #
            #     arguments = []
            #     argument_count += len(event_mention['arguments'])
            #     for argument in event_mention['arguments']:
            #         position = argument['position']
            #         start_idx, end_idx = find_token_index(
            #             tokens=tokens,
            #             start_pos=position[0] - sent_start_pos,
            #             end_pos=position[1] - sent_start_pos + 1,
            #             phrase=argument['text'],
            #         )
            #
            #         argument['start'] = start_idx
            #         argument['end'] = end_idx
            #         del argument['position']
            #
            #         arguments.append(argument)
            #
            #     event_mention['arguments'] = arguments
            #     data['golden-event-mentions'].append(event_mention)

            # result.append(data)
            sentences.append(data)
        result.append(doc)

    # print('======[Statistics]======')
    # print('sent :', sent_count)
    # print('event :', event_count)
    # print('entity :', entity_count)
    # print('argument:', argument_count)

    verify_result(result)
    # with open('output/{}.json'.format(data_type), 'w') as f:
    #     json.dump(result, f, indent=2)

    # transfer into dygie json format
    entity_count, sent_count, doc_count = 0, 0, 0
    stats_treebank = dict(sent_mismatches=0, sent_matches=0)
    fp = open('output/{}.json'.format(data_type), 'w')
    for r in result:
        dygie_doc = {}
        doc_count += 1
        dygie_doc['doc_key'] = r['name']+'_'+data_type
        dygie_doc['sentences'] = []
        dygie_doc['ner'] = []
        dygie_doc['trees'] = []
        offset = 0
        for sentence in r['sentences']:
            dygie_doc['sentences'].append(sentence['words'])
            sent_count += 1
            entity_of_this_sentence = []
            for entity in sentence['golden-entity-mentions']:
                dygie_entity = []
                entity_count += 1
                dygie_entity.append(entity['start']+offset)
                dygie_entity.append(entity['end']+offset-1) # inclusive endpoint
                dygie_entity.append(entity['type'])
                entity_of_this_sentence.append(dygie_entity)
            dygie_doc['ner'].append(entity_of_this_sentence)
            offset += len(sentence['words'])

            tree, _ = readTree(sentence['parse'], 0)
            nodes = []
            get_node(tree, nodes, -1)
            tree = Tree(nodes)
            tree.get_span_for_leaf_node(sentence['words'])
            if not tree.match:
                print(
                    "sent mismatch, doc: {}, original sentence: {}, tree sentence {}".format(
                        r['name'], sentence['words'], tree.show_leaf_node()))
                stats_treebank['sent_mismatches'] += 1
            else:
                stats_treebank['sent_matches'] += 1
            tree.get_span_for_node(sentence['words'])
            dygie_doc['trees'].append(tree.to_json())


        fp.write(json.dumps(dygie_doc) + "\n")

    print('======[Statistics]======')
    print('doc :', doc_count)
    print('sent :', sent_count)
    print('entity :', entity_count)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help="Path of ACE2005 English data", default='./data/ace_2005_td_v7/data/English')
    args = parser.parse_args()

    # generate_data_list(args.data, './data_list_fei.csv')

    test_files, dev_files, train_files = get_data_paths(args.data, './data_list_fei.csv')

    with StanfordCoreNLP('./stanford-corenlp-full-2018-10-05', memory='8g', timeout=60000) as nlp:
        # res = nlp.annotate('Donald John Trump is current president of the United States.', properties={'annotators': 'tokenize,ssplit,pos,lemma,parse'})
        # print(res)

        # preprocessing('dev', dev_files)
        preprocessing('test', test_files)
        # preprocessing('train', train_files)
