from transformers import logging
import tqdm
import torch
import re
import json
from statistics import mean
from sklearn.metrics import f1_score
from nltk import sent_tokenize

logging.set_verbosity_error()


class Evaluator:
    def __init__(
        self,
        model,
        model_id,
        tokenizer,
        eval_dataset,
        task,
        schema,
        cuda_device,
        max_new_tokens=512,
    ):
        self.model = model
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.eval_dataset = eval_dataset
        self.task = task
        self.schema = schema
        self.device = cuda_device
        
        

        self.struc_test_mapping = {
            "XML": {
                "acl_arc": [
                    (self.is_xml, {"pattern":r'#TARGET_REF<(?:BACKGROUND|USE|COMPARISON|MOTIVATION|EXTENSION|FUTURE)/>'}),
                    (self.has_right_tags, {"pattern":r'<(?:BACKGROUND|USE|COMPARISON|MOTIVATION|EXTENSION|FUTURE)/>', "requirements":[('self_closing','eq',1)]}),
                    (self.is_similar_to_input, {"pattern":r'<(?:BACKGROUND|USE|COMPARISON|MOTIVATION|EXTENSION|FUTURE)/>'}),
                ],
                "finecite": [
                    (self.is_xml, {"pattern":r'</?(?:INFORMATION|PERCEPTION|BACKGROUND)>'}),
                    (self.has_right_tags, {"pattern":r'</?(?:INFORMATION|PERCEPTION|BACKGROUND)>', "requirements":[('opening','ge',1),('closing','ge',1)]}),
                    (self.is_similar_to_input, {"pattern":r'</?(?:INFORMATION|PERCEPTION|BACKGROUND)>'}),

                ],
                "multicite": [
                    (self.is_xml, {"pattern":r'#TARGET_REF<(?:BACKGROUND|USE|DIFFERENCE|SIMILARITY|MOTIVATION|EXTENDS|FUTURE)/>|</?CONT>'}),
                    (self.has_right_tags, {"pattern":r'<(?:BACKGROUND|USE|DIFFERENCE|SIMILARITY|MOTIVATION|EXTENDS|FUTURE)/>|</?CONT>', "requirements":[('self_closing','ge',1),('opening','ge',1),('closing','ge',1)]}),
                    (self.is_similar_to_input, {"pattern":r'<(?:BACKGROUND|USE|DIFFERENCE|SIMILARITY|MOTIVATION|EXTENDS|FUTURE)/>|</?CONT>'}),
                ]
            },
            "JSON": {
                "acl_arc": [
                    (self.is_json, None),
                    (self.has_keys, {"keys": ['label']}),
                    (self.has_type, {"key_type_mapping": [('label',list)]}),
                    (self.class_exists, {"key":"label", "expected_classes":['BACKGROUND','USE','COMPARISON','MOTIVATION','EXTENSION','FUTURE']}),
                ],
                "finecite": [
                    (self.is_json, None),
                    (self.has_keys, {"keys":['INFORMATION','PERCEPTION','BACKGROUND']}),
                    (self.has_type, {"key_type_mapping":[('INFORMATION',list),('PERCEPTION',list),('BACKGROUND',list)]}),
                    (self.is_similar_to_input, {'key':'INFORMATION'}),
                    (self.is_similar_to_input, {'key':'PERCEPTION'}),
                    (self.is_similar_to_input, {'key':'BACKGROUND'}),
                ],
                "multicite": [
                    (self.is_json, None),
                    (self.has_keys, {"keys":['label','context']}),
                    (self.has_type, {"key_type_mapping":[('label',list),('context',list)]}),
                    (self.class_exists, {"key":"label", "expected_classes":['BACKGROUND','MOTIVATION','USE','EXTENDS','SIMILARITY','DIFFERENCE','FUTURE']}),
                    (self.is_similar_to_input, {'key':'context'}),
                ]
            },
            # "JSON2": {
            #     "acl_arc": [
            #         (self.is_json, None),
            #         (self.has_keys, {"keys":['label']}),
            #         (self.has_type, {"key_type_mapping":[('label',list)]}),
            #         (self.class_exists, {"key":"label", "expected_classes":['BACK','USE','COMPARE_CONTRAST','MOT','EXTENSION','FUT']}),
            #     ],
            #     "finecite": [
            #         (self.is_json, None),
            #         (self.has_keys, {"keys":['INFO','PERCEPT','BACK']}),
            #         (self.has_type, {"key_type_mapping":[('INFO',list),('PERCEPT',list),('BACK',list)]}),
            #         (self.class_exists, {"key":"INFO", "expected_classes":'sent_classes'}),
            #         (self.class_exists, {"key":"PERCEPT", "expected_classes":'sent_classes'}),
            #         (self.class_exists, {"key":"BACK", "expected_classes":'sent_classes'}),
            #     ],
            #     "multicite": [
            #         (self.is_json, None),
            #         (self.has_keys, {"keys":['label','context']}),
            #         (self.has_type, {"key_type_mapping":[('label',list),('context',list)]}),
            #         (self.class_exists, {"key":"label", "expected_classes":['BACK','MOT','USE','EXT','SIM','DIFFER','FUT']}),
            #         (self.class_exists, {"key":"context", "expected_classes":'sent_classes'}),
            #     ]
            # }
        }

        self.qual_eval_mapping = {
            "extraction": {
                "XML": {
                    "acl_arc": [(self.extract_classes,{'pattern':r'<(BACKGROUND|USE|COMPARISON|MOTIVATION|EXTENSION|FUTURE)/>'},'label')],
                    "finecite": [(self.get_token_context,{'patterns':[r'<INFORMATION>(.+?)</INFORMATION>',r'<PERCEPTION>(.+?)</PERCEPTION>',r'<BACKGROUND>(.+?)</BACKGROUND>']}, 'context')],
                    "multicite": [
                        (self.extract_classes,{'pattern':r'<(BACKGROUND|USE|DIFFERENCE|SIMILARITY|MOTIVATION|EXTENDS|FUTURE)/>'},'label'),
                        (self.get_sent_context,{'patterns':[r'<CONT>(.+?)</CONT>']}, 'context')
                    ]
                },
                "JSON": {
                    "acl_arc": [(self.extract_classes,{'key':'label'},'label')],
                    "finecite": [(self.get_token_context,{'keys':['INFORMATION','PERCEPTION','BACKGROUND']}, 'context')],
                    "multicite": [
                        (self.extract_classes,{'key':'label'}, 'label'),
                        (self.get_sent_context,{'key':'context'}, 'context')
                    ]
                },
                # "JSON2": {
                #     "acl_arc": [(self.extract_classes,{'key':'label'},'label')],
                #     "finecite": [(self.get_sent_context, {'keys':['INFO','PERCEPT','BACK']}, 'context')],
                #     "multicite": [
                #         (self.extract_classes,{'key':'label'}, 'label'),
                #         (self.get_sent_context,{'keys':['context']}, 'context')
                #     ]
                # }
            },
            "evaluation": {
                "acl_arc": [(self.calculate_micro_macro, {'key':'label','mapping':{'BACKGROUND':0,'USE':1,'COMPARISON':2,'MOTIVATION':3,'EXTENSION':4,'FUTURE':5}})],
                "finecite": [
                    (self.calculate_micro_macro, {'key':'context'}),
                    (self.metrics_for_finecite, {'key':'context'}),
                    ],
                "multicite": [
                    (self.weak_stron_accuracy, {'key':'label', 'mapping':{'BACKGROUND':0,'USE':1,'DIFFERENCE':2,'SIMILARITY':3,'MOTIVATION':4,'EXTENDS':5,'FUTURE':6}}),
                    (self.calculate_micro_macro, {'key': 'context'})
                ]
            }
        }

        model.to(self.device)

    def evaluate(self, test_data=None):
        input_ids = self.eval_dataset['input_ids']
        gold_annotation = self.eval_dataset['gold']
        task_strs = self.eval_dataset['input']


        #generate task completion
        if test_data:
            pred_annotation = test_data
        else:
            pred_annotation = self.generate(input_ids)

        #evaluate structure
        is_valid_annotation = self.evaluate_structure(task_strs, pred_annotation)

        # print(task_strs)
        # print(pred_annotation)
        # print(is_valid_annotation)

        #extract results from valid entries
        val_samples = [(gold, preds) for gold, preds, valid in zip(gold_annotation, pred_annotation, is_valid_annotation) if valid]
        clean_samples = self.extractor(val_samples)
        #print(clean_samples)

        # evaluate qulaity
        qual_eval = self.evaluate_quality(clean_samples, is_valid_annotation)
        print(qual_eval)
        return qual_eval

    def generate(self, input_ids):
        pred_annotation = []
        for ids in tqdm.tqdm(input_ids[:150]):
            # print(self.tokenizer.decode(ids).split('[INST]')[-1])
            res = self.model.generate(torch.tensor([ids]).to(self.device), max_new_tokens = self.max_new_tokens)
            if self.model_id == 'mistral':
                output = self.tokenizer.decode(res[0]).split('[/INST]')[-1] # needs to be adjusted for other models .split('<|end_header_id|>')[-1] #
                output = re.sub('</s>', '', output)# needs to be adjusted for other models re.sub('\n|<\|eot_id\|>', '', output) #
            if self.model_id == 'llama':
                output = self.tokenizer.decode(res[0]).split('<|end_header_id|>')[-1] #
                output = re.sub('\n|<\|eot_id\|>', '', output) #
            pred_annotation.append(output)
            print(output)
        return pred_annotation

    def evaluate_structure(self, task_ins, task_outs):
        structure_is_valid = []
        for t_in, t_out in zip(task_ins, task_outs):
            task_desc = self.struc_test_mapping[self.schema][self.task]
            try:
                for eval_task, kwargs in task_desc:
                    if kwargs:
                        res = eval_task(t_in=t_in, t_out=t_out, **kwargs)
                    else:
                        res = eval_task(t_in=t_in, t_out=t_out)
                    if res == False:
                        structure_is_valid.append(False)
                        raise
            except:
                function = re.search(r'(?<=Evaluator\.)[^\s]+', str(eval_task)).group()
                print(f"Structure failed at function: {function if function else 'unknown function'}, {kwargs}")
                continue
            structure_is_valid.append(True)
        return structure_is_valid

    def extractor(self, samples):
        clean_sample = []
        for sample_g, sample_p in samples:
            clean_pred = {}
            for task, kwargs, key in self.qual_eval_mapping['extraction'][self.schema][self.task]:
                if kwargs:
                    res = task(gold=sample_g, pred=sample_p, **kwargs)
                else:
                    res= task(gold=sample_g, pred=sample_p)
                clean_pred[key] = res
            clean_sample.append((sample_g, clean_pred))
        return clean_sample

    def evaluate_quality(self, samples, is_valid):
        eval_metrics = {}

        #add structural integrity eval
        eval_metrics['struc_acc'] = mean(is_valid)

        #calculate all other metrics
        for task, kwargs in self.qual_eval_mapping['evaluation'][self.task]:
            if kwargs:
                res = task(samples=samples, **kwargs)
            else:
                res= task(samples=samples)
            for key, val in res:
                eval_metrics[key] = val

        return eval_metrics


    #string SIM
    def minimum_word_difference(self, s1:list, s2:list):
        n_s1, n_s2 = len(s1)+1, len(s2)+1
        gap_p, mis_p = 1, 1

        M = [[float('inf') for _ in range(n_s1)] for _ in range(n_s2)]
        for i in range(n_s1):
            M[0][i] = 0
        for j in range(n_s2):
            M[j][0] = round(j * gap_p, 2)

        for i in range(1,n_s2):
            for j in range(1,n_s1):
                M[i][j] = round(min(
                    M[i-1][j-1] if s1[j-1] == s2[i-1] else mis_p + M[i-1][j-1],
                    gap_p + M[i-1][j],
                    gap_p + M[i][j-1]
                ), 2)

        #get minimal cost and index of its last coordinates
        min_cost = min(M[n_s2-1])
        min_index_last_row = next(i for i in reversed(range(len(M[n_s2-1]))) if M[n_s2-1][i] == min_cost)

        # traceback minimum editing distance
        string_match = self.traceback(M, n_s2-1, min_index_last_row, [])

        # get string overlap
        overlap = self.get_overlap(string_match, [])

        return {"SIM": 1 - min_cost/len(overlap), "overlap": overlap}

    def traceback(self, M, i, j, string_match):
        string_match.insert(0, (i,j))
        if i==0 and j== 0: return string_match
        elif i==0 and j > 0: i, j = i,j-1
        elif j==0 and i > 0: i,j = i-1, j
        elif j > 0 and i > 0:
            idx = [(i-1, j-1), (i-1, j), (i, j-1)]
            cost = float('inf')
            for _i, _j in idx:
                if M[_i][_j] < cost:
                    cost = M[_i][_j]
                    i, j = _i, _j
        res = self.traceback(M, i, j, string_match)
        return res

    def get_overlap(self, match, overlap):
        m = match.pop(0)
        if m[0] != 0:
            overlap.append(m)
        if match and m[0] != match[-1][0]:
            overlap = self.get_overlap(match, overlap)
        return overlap


    # structual evaluation
    def is_json(self, t_out, **kwargs):
        try:
            json.loads(t_out)
        except ValueError as e:
            return False
        return True

    def has_keys(self, t_out, keys, **kwargs):
        json_dict = json.loads(t_out)
        for key in keys:
            if key not in json_dict.keys():
                return False
        return True

    def has_type(self, t_out, key_type_mapping, **kwargs):
        json_dict = json.loads(t_out)
        for key, exp_type in key_type_mapping:
            if type(json_dict[key]) != exp_type:
                return False
        return True

    def class_exists(self, t_in, t_out, key, expected_classes, **kwargs):
        json_dict = json.loads(t_out)

        if expected_classes == 'sent_classes':
            expected_classes = re.findall(r'sent\d{1,2}', t_in)

        for cls in json_dict[key]:
            if cls not in expected_classes:
                return False
        return True

    def is_similar_to_input(self, t_in, t_out, key = None, pattern = None, **kwargs):
        str_in = t_in.split()

        #make json from input
        if key:
            str_out = json.loads(t_out)
            str_out = [text.split() for text in str_out[key]]
            if str_out == []: return True

        #strip xml tags from input
        if pattern:
            str_out = [re.sub(pattern, '', t_out).split()]
            # print(str_in)
            # print(str_out[0])
            if abs(len(str_out[0]) - len(str_in)) > len(str_in) * 0.1: return False

        #assess SIM of output string to input string
        SIM = mean([self.minimum_word_difference(str_in, out_text)['SIM'] for out_text in str_out])
        #print(SIM)
        if SIM < 0.9: return False
        return True

    def is_xml(self, t_out, pattern, **kwargs):
        tags = re.findall(pattern, t_out)
        if tags == []: return False
        opened_tag = ''
        for tag in tags:
            if re.match(r'<[^\/]+?>', tag):
                if opened_tag != '':return False
                opened_tag = re.match(r'<([^\/]+?)>', tag).group(1)
            elif re.match(r'</.+?>', tag):
                closing_tag = re.match(r'</(.+?)>', tag).group(1)
                if closing_tag != opened_tag: return False
                opened_tag = ''
        return True

    def has_right_tags(self, t_out, pattern, requirements, **kwargs):
        tags = re.findall(pattern, t_out)
        stats = {}
        stats['opening'] = re.findall(r'<[^\/]+?>', ''.join(tags))
        stats['closing'] = re.findall(r'</.+?>', ''.join(tags))
        stats['self_closing'] = re.findall(r'<[^>]+?/>', ''.join(tags))
        for key, method, req in requirements:
            if method == 'eq':
                if len(stats[key]) != req:
                    return False
            if method == 'ge':
                if len(stats[key])  < req:
                    print(stats)
                    print(t_out)
                    return False
        return True

    #extractor functions
    def extract_classes(self, pred, key=None, pattern=None, **kwargs):
        #make json from input
        if key:
            json_dict = json.loads(pred)
            return json_dict[key]

        #strip xml tags from input
        if pattern:
            return re.findall(pattern, pred)

    def get_sent_context(self, gold, pred, key = None, keys=None, patterns=None, **kwargs):
        context_arr = [0 for _ in range(len(gold['context']))]

        if keys:
            json_dict = json.loads(pred)
            for i, k in enumerate(keys):
                indices = [int(sent[4:]) for sent in json_dict[k]]
                for index in indices:
                    context_arr[index] = i+1

        if key:
            json_dict = json.loads(pred)
            context_sent_probs = [[self.minimum_word_difference(text.split(), out_text.split())['SIM'] for text in gold['text']] for out_text in json_dict[key]]
            for sent_probs in context_sent_probs:
                idx = sent_probs.index(max(sent_probs))
                context_arr[idx] = 1

        if patterns:
            context_texts = [re.sub(r'<.+?/>', '',out_text) for out_text in re.findall(patterns[0], pred)]
            context_sent_probs = [[self.minimum_word_difference(out_text.split(), text.split())['SIM'] for text in gold['text']] for out_text in context_texts]
            for sent_probs in context_sent_probs:
                for idx, prob in enumerate(sent_probs):
                    if prob > 0.85:
                        context_arr[idx] = 1

        # print(context_arr)
        # print(gold['context'])
        return context_arr

    def get_token_context(self, gold, pred, keys = None, patterns=None, **kwargs):
        text = gold['text']
        #print(gold)
        context_arr = [0 for _ in range(len(gold['context']))]

        if keys:
            json_dict = json.loads(pred)
            out_texts = [(out_text, i) for i, key in enumerate(keys) for out_text in json_dict[key]]

        if patterns:
            out_texts = [(out_text, i) for i, pattern in enumerate(patterns) for out_text in re.findall(pattern, pred)]
        #print(out_texts)
        for out_text, i in out_texts:
            overlap = self.minimum_word_difference(text, out_text.split())['overlap']
            for _, index in overlap:
                context_arr[index-1] = i + 1

        return context_arr


    #evaluation function
    def calculate_micro_macro(self, samples, key, mapping=None):
        if key == 'label':
            gold = [mapping[sample[0][key][0]] for sample in samples]
            preds = [mapping[sample[1][key][0]] for sample in samples]
            print(gold,preds)
            macro_f1 = f1_score(gold,preds, average='macro')
            micro_f1 = f1_score(gold,preds, average='micro')

        if key == 'context':
            gold = [sample[0][key] for sample in samples]
            preds = [sample[1][key] for sample in samples]

            #print(list(zip(gold, preds)))

            macro_f1 = mean([f1_score(g, p, average='macro') for g, p in zip(gold, preds)])
            micro_f1 = mean([f1_score(g, p, average='micro') for g, p in zip(gold, preds)])

        return ('micro', float(micro_f1)), ('macro', float(macro_f1))

    def weak_stron_accuracy(self, samples, key, mapping=None):
        if key == 'label':
            gold = [[mapping[class_entry] for class_entry in sample[0][key]] for sample in samples]
            preds = [[mapping[class_entry] for class_entry in sample[1][key]] for sample in samples]

            weak_acc = []
            strict_acc = []

            for g, p in zip(gold, preds):
                g, p = sorted(g), sorted(p)
                print(list(zip(g,p)))
                if all([i==j for i, j in zip(g, p)]): strict_acc.append(True)
                else: strict_acc.append(False)
                if any([i in g for i in p]): weak_acc.append(True)
                else: weak_acc.append(False)

        return ('weak_acc', mean(weak_acc)), ('strict_acc', mean(strict_acc))


    def metrics_for_finecite(self, samples, key):
        gold = [sample[0][key] for sample in samples]
        preds = [sample[1][key] for sample in samples]

        total_gold = [[0 if value==0 else 1 for value in sample] for sample in gold]
        total_preds = [[0 if value==0 else 1 for value in sample] for sample in preds]
        total_f1 = mean([f1_score(g, p) if sum(g) != 0 or sum(p) != 0 else 1 for g, p in zip(total_gold, total_preds)])

        inf_gold = [[1 if value==1 else 0 for value in sample] for sample in gold]
        inf_preds = [[1 if value==1 else 0 for value in sample] for sample in preds]
        inf_f1 = mean([f1_score(g, p) if sum(g) != 0 or sum(p) != 0 else 1 for g, p in zip(inf_gold, inf_preds)])

        perc_gold = [[1 if value==2 else 0 for value in sample] for sample in gold]
        perc_preds = [[1 if value==2 else 0 for value in sample] for sample in preds]
        perc_f1 = mean([f1_score(g, p) if sum(g) != 0 or sum(p) != 0 else 1 for g, p in zip(perc_gold, perc_preds)])

        back_gold = [[1 if value==3 else 0 for value in sample] for sample in gold]
        back_preds = [[1 if value==3 else 0 for value in sample] for sample in preds]
        back_f1 = mean([f1_score(g, p) if sum(g) != 0 or sum(p) != 0 else 1 for g, p in zip(back_gold, back_preds)])

        return ('total_f1', float(total_f1)), ('inf_f1', float(inf_f1)), ('perc_f1', float(perc_f1)), ('back_f1', float(back_f1))
