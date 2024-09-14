from transformers import logging
from tqdm import tqdm
import torch
import re
import json


logging.set_verbosity_error()


class Evaluator:
    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens,
        eval_dataset,
        task,
        schema,
        cuda_device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.eval_dataset = eval_dataset
        self.task = task
        self.schema = schema
        self.device = cuda_device
        
        self.struc_test_mapping = {
                "JSON1": {
                    "tests": [self.is_json, self.has_keys, self.class_exists],
                    "kwargs": {
                        "acl_arc": [None, {"keys":['label']},{"key":"label","expected_classes":['BACKGROUND','MOTIVATION','COMPARE_CONTRAST','USE']}]
                        }
                    }
        }
        
        model.to(self.device)

    def evaluate(self):  
        #generate task completion
        annotations = self.generate()
        print(annotations)
        #evaluate structure
        struc_eval = self.evaluate_structure(annotations)
        
        # # add to predictions if ok
        # if struc_eval: 
        #     json_dict = json.loads(output)
        #     predictions.append({
        #         'gold':gold_label,
        #         'pred': json_dict['label']
        #     })
            
        # #qualitative eval
        # qual_eval = self.evaluate_quality(predictions)
        # print(qual_eval)
    
    def generate(self):
        completions = []
        for sample in tqdm(self.eval_dataset):
            input_ids = sample['input_ids']  
            input_ids = torch.tensor([input_ids]).to(self.device) 
            res = self.model.generate(input_ids, max_new_tokens = self.max_new_tokens)
            output = self.tokenizer.decode(res[0]).split('[/INST]')[-1] # needs to be adjusted for other models
            output = re.sub('</s>', '', output) # needs to be adjusted for other models
            completions.append(output)
        return completions
    
    def evaluate_structure(self, completions):
        structure_is_valid = []
        for comp in completions:
            task_desc = self.struc_test_mapping[self.schema]
            for task, kwargs in zip(task_desc['tests'], task_desc['kwargs'][self.task]):
                if kwargs:
                    res = task(comp, **kwargs)
                else:
                    res = task(comp)
                if res == False:
                    structure_is_valid.append(False)
                    break
            structure_is_valid.append(True)         
        return structure_is_valid
        
    
    def is_json(self, json_str):
        try:
            json.loads(json_str)
        except ValueError as e:
            return False
        return True
    
    def has_keys(self, json_str, keys):
        json_dict = json.loads(json_str)
        for key in keys:
            if key not in json_dict.keys():
                return False
        return True
    
    def class_exists(self, json_str, key, expected_classes):
        json_dict = json.loads(json_str)
        for cls in json_dict[key]:
            print(cls)
            if cls not in expected_classes:
                return False
        return True
    
    def evaluate_quality(self, preds):
        accuracy = mean([1 if sample['gold'] == sample['pred'] else 0 for sample in preds])
        return accuracy
    


def calculate_metrics(eval_df):
    valid_json = sum(eval_df['is_valid_json']) / len(eval_df)
    eval_df.dropna(inplace=True)
    macro_f1 = f1_score([int(no) for no in eval_df['label']], [int(no) for no in eval_df['label_pred']], average='macro')
    micro_f1 = f1_score([int(no) for no in eval_df['label']], [int(no) for no in eval_df['label_pred']], average='micro')
    return valid_json, micro_f1, macro_f1