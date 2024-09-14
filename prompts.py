prompt_building_blocks = {
    "prov_mat": {
        "v1": "You are given a excerpt from a scientific text with one citation marked as #TARGET_REF and all other marked with #REF. Further you are given a list of citation function classes."
    },
    "task_desc":{
        "v1": "Your task is it to find the citation function class of the marked citation."
    },
    "chain_o_thought": {
        "v1": "To do this, first, find the targeted citation, second, identify which function class fits the best for the marked citation by considering the surrounding words, third, reflect on your label choice until you are sure."
    },
    "outp_desc": {
        "v1": "Reply with a short valid json only containing the label e.g.: {\"label\": [\"USE\"]}."
    },
    "further_inst": {
        "v1": "Make sure the output is a valid jason with "" instead of \'\''"
    },
    "list_o_labels": {
        "v1": f"""\n\nList Citation Function Types\n
                BACKGROUND: The cited paper provides relevant background information or is part of the body of literature. \n
                USE: The citing paper uses the methodology or tools created by the cited paper. \n
                COMPARE_CONTRAST: The citing paper expresses similarities to or or differences from, or disagrees with, the cited paper.\n
                MOTIVATION: The citing paper is directly motivated by the cited paper. \n
                EXTENSION: The citing paper extends the methods, tools, or data of the cited paper.\n
                FUTURE: The cited paper is a potential avenue for future work. \n
                """
    },
    "reassur": {
        "v1": "Are the instructions clear to you?"
    },
    "model_response": {
        "v1": "Yes, the instructions are clear to me. I will determine the citation function class of the marked citation (#AUTHOR_TAG) based on the provided citation function types and respond in json fromat."
    },
    "ex_task": {
        "v1": "In a similar vain to #AUTHOR_TAG and Buchholz et al. ( 1999 ) , the method extends an existing flat shallow-parsing method to handle composite structures ."
    },
    "ex_ann":{
        "v1": "{\"label\": [\"COMPARE_CONTRAST\"]}"
    },
}

promt_mapping = {
    "acl_arc": {
        "JSON1": [
            {
                "role": "user",
                "content": [("prov_mat","v1"),("task_desc","v1"),("chain_o_thought","v1"),("outp_desc","v1"),("further_inst","v1"),("list_o_labels","v1"),("reassur","v1")],
            },
            {
                "role": "assistant",
                "content": [("model_response","v1")]
            },
            {
                "role": "user",
                "content": [("ex_task","v1")]
            },
            {
                "role": "assistant",
                "content": [("ex_ann","v1")]
            }
        ]
    }
}

class PromptForAutoCCA:
    
    def __init__(
        self,
        tokenizer,
        task,
        schema,
    ):
        self.tokenizer = tokenizer
        self.task = task
        self.schema = schema
        
    def create_sample(self, input_str, label_str, for_generation = False):
        prompt = self.get_prompt()
        
        #add input_str
        prompt.append({"role": "user", "content": input_str})
        
        #add label_str if not for_generation
        if not for_generation: prompt.append({"role": "assistant", "content": label_str})
        encoded_input_ids = self.tokenizer.apply_chat_template(prompt)

        return {'input_ids': encoded_input_ids}

    def get_prompt(self):
        mappings = promt_mapping[self.task][self.schema]
        messages = []
        for mapping in mappings:
            role = mapping['role']
            content = [prompt_building_blocks[block][version] for block, version in mapping['content']]
            msg = {
                "role": role,
                "content": ' '.join(content)
            }
            messages.append(msg)
        return messages