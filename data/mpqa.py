"""
Read raw mpqa data.
"""

import os
import re
import glob
from typing import Any, Dict, List, Union


REGEX_ATTR = re.compile(r'[a-z\-]*\=\"[^"]*\"')


def read_mpqa_doc(doc_path: str) -> List[Dict[str, Any]]:
    def in_which_sentence(span) -> int:
        for i, sent in enumerate(sentences):
            if span[0] >= sent['span'][0] and span[1] <= sent['span'][1]:
                return i, sent['span'][0]
        return -1, 0

    man_path = doc_path.replace('docs', 'man_anns') + '/gateman.mpqa.lre.2.0'
    sent_path = man_path.replace('gateman.mpqa.lre', '/gatesentences.mpqa')
    with open(doc_path) as d, open(man_path) as m, open(sent_path) as s:
        doc = d.read()
        ann_lines = [r.strip().split('\t') for r in m.readlines() if not r.startswith('#')]
        sentence_lines = [r.split() for r in s.readlines()]

    doc_id = '_'.join(doc_path.split('/')[-2:])
    sentences = []
    for line in sentence_lines:
        if line[-1] != 'GATE_sentence':
            continue
        span = tuple(int(s) for s in line[1].split(','))
        raw = doc[span[0]: span[1]]
        sentence = dict(id=f"{doc_id}_{line[0]}", span=span, raw=raw, ann=list())
        sentences.append(sentence)

    for line in ann_lines:
        span = tuple(int(s) for s in line[1].split(','))
        index, offset = in_which_sentence(span)
        if index == -1:
            continue
        inner_span = tuple(s - offset for s in span)
        ann_type = line[3].replace('GATE_', '')
        ann = dict(type=ann_type, span=inner_span)
        if len(line) >= 5:
            attr = [a.split('=') for a in REGEX_ATTR.findall(line[4])]
            attr = {a[0]: a[1].replace('"', '').replace(', ', ',') for a in attr}
            ann.update(RT=line[4], **attr)
        sentences[index]['ann'].append(ann)

    for s in sentences:
        s.pop('span')

    return sentences


def read_mpqa(mpqa_dir: str) -> List[Dict[str, Union[str, List[Dict]]]]:
    doc_pathes = glob.glob(os.path.normpath(mpqa_dir + '/docs/*/*'))
    sentences = list()
    for path in doc_pathes:
        sentences.extend(read_mpqa_doc(path))
    return sentences

###############################################
# The following part is not finished.
# 下面没写完, 暂时没需求


attr_set, C = set(), 0
INVALID_PAIRS = []
BAD_AGENT, BAD_TARGET, BAD_EXP = [], [], []


def mpqa_to_opinion(path: str) -> List[Dict[str, Any]]:
    data = list()
    global C

    def get_source_ids(string: str):
        source_ids = [s for s in string.split(',') if s]
        if 'w' in source_ids:
            source_ids.remove('w')
            # if len(source_ids) > 1:
            #     print(0)
        return source_ids

    for ins in read_mpqa(path):
        expressions, agents, targets = dict(), dict(), dict()
        for ann in ins['ann']:
            if ann['type'] == 'direct-subjective':
                if 'polarity' not in ann or ann['polarity'] not in ('positive', 'negative'):
                    # print(ann)
                    continue
                # assert ann['span'] not in expressions
                if ann['span'] in expressions:
                    ori = expressions[ann['span']]
                    if (ori_source := ori['nested-source']) != (now_source := ann['nested-source']):
                        print(ori_source)
                        print(now_source)
                        print(0)
                    if (ori_polarity := ori['polarity']) != (now_polarity := ann['polarity']):
                        print(ori_polarity)
                        print(now_polarity)
                        print(0)
                    if 'attitude-link' in expressions[ann['span']]:
                        continue
                    # elif ann['type'] == 'expressive-subjectivity':
                    #     continue
                # label = 'POS' if ann['polarity'] == 'positive' else 'NEG'
                # label = ann['polarity']
                # if 'nested-source' not in ann:
                #     BAD_EXP.append(ann)
                #     continue
                # source_ids = get_source_ids(ann['nested-source'])
                # 只有 ann['type'] == 'direct-subjective' 时有 'attitude-link'
                # exp = dict(span=ann['span'], label=label, source_ids=ann['nested-source'])
                # if ann['type'] == 'direct-subjective' and 'attitude-link' not in ann:
                #     breakpoint()  # 就是存在找不到target的
                # if 'attitude-link' in ann:
                #     exp.update(link=ann['attitude-link'])
                #     attr_set.add(ann['attitude-link'])
                expressions[ann['span']] = ann
            # if 'attitude-type' in ann and 'sentiment' in ann['attitude-type']:
            #     # 只有 ann['type'] == 'attitude' 时有 'attitude-type
            #     if 'id' in ann:
            #         attitudes[ann['id']] = ann
            #     else:  # 只有两个 sentiment 是 inferred, 来自两个不同的句子
            #         attitudes['inferred'] = ann
            elif ann['type'] == 'agent':
                if 'id' in ann:
                    agents[ann['id']] = ann
                elif 'nested-source' in ann:
                    _ann_id = ann['nested-source'].split(',')[-1]
                    agents[_ann_id] = ann
                else:
                    BAD_AGENT.append(ann)
            elif ann['type'] == 'target':
                if 'id' in ann:
                    targets[ann['id']] = ann
                elif 'nested-source' in ann:
                    _ann_id = ann['nested-source'].split(',')[-1]
                    targets[_ann_id] = ann
                else:
                    BAD_TARGET.append(ann)

        if expressions:
            pass
            # print(0)
            # elif 'attitude-link' in ann:
            #     if 'nested-source' not in ann:
            #         BAD_DIRECT.append(ann)
            #         continue
            #     source_ids = [s for s in ann['nested-source'].split(',') if s]
            #     if 'w' in source_ids:
            #         source_ids.remove('w')
            #         # if len(source_ids) > 1:
            #         #     print(0)
            #     attitude_ids = [s for s in ann['attitude-link'].split(',') if s]
            #     for i in attitude_ids:
            #         for a in ins['ann']:
            #             if 'id' in a and a['id'] == i:
            #                 if 'intensity' not in a:
            #                     C += 1
            #                 for k in ('polarity', 'intensity', 'expression-intensity'):
            #                     if k in a:
            #                         attr_set.add((k, a[k]))
            #     pairs |= set(product(source_ids, attitude_ids))

        # if attitudes:
        #     opinions, holders = dict(), dict()
        #     pairs = [p for p in pairs if p[1] in attitudes and p[0] != 'implicit']
        #     for h, _ in pairs:
        #         if h in agents:
        #             holders[h] = agents[h]['span']
        #         else:
        #             INVALID_PAIRS.append((h, _))
        #     for k, v in attitudes.items():
        #         opinions[k] = [v['attitude-type'][-3:].upper(), *v['span']]
        #     data.append(dict(id=ins['id'], raw=ins['raw'], opinions=opinions, holders=holders, pairs=pairs))

    return data
