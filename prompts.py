prompt_building_blocks = {
    "provided_material": {
        "v1": "You are given a list of citation function classes and an excerpt from a scientific text in which citations are marked #REF and #TARGET_REF.",
        "v2": "You are given a list of citation context classes and an excerpt from a scientific text in which citations are marked #REF and #TARGET_REF.",
        # "v3": "You are given a list of citation function classes and an list of sentences from a scientific text in which citations are marked #REF and #TARGET_REF.",
        # "v4": "You are given a list of citation context classes and an list of sentences from a scientific text in which citations are marked #REF and #TARGET_REF.",
    },
    "task_desc":{
        "v1": "Assign the correct function classes to the #TARGET_REF.",
        "v2": "Extract the citation context of the #TARGET_REF using the context classes",
        "v3": "Assign correct function classes and extract the citation context of the #TARGET_REF",
    },
    "chain_of_thought": {
        "v1": "To do this, first, locate #TARGET_REF, second, select the best fitting function class by considering the target citation context, third, reflect on whether you made the correct choice and respond when you are sure it is correct.",
        "v2": "To do this, first, locate #TARGET_REF, second, identify the target citation context and assign the context classes, third, third, reflect on whether you made the correct choice and respond when you are sure it is correct.",
        "v3": "To do this, first, locate #TARGET_REF, second, identify the target citation context, third, select the best fitting function classes, fourth, reflect on whether you made the correct choice and respond when you are sure it is correct.",
    },
    "output_desc": {
        "v1": "Reply by adding a self closing XML tag containing the right function class directly after the #TARGET_REF (e.g: #TARGET_REF<USE/>)",
        "v2": "Reply with a valid JSON containing the function label (e.g: {\"label\": [\"USE\"]})",
        "v3": "Reply by surrounding context sequences wiht XML tags containing the right context class (e.g: <BACKGROUND>...</BACKGROUND>)",
        "v4": "Reply with a valid JSON containing the context sequences for each context class (e.g: {\"INFORMATION\": [\"...\"],\"PERCEPTION\": [\"...\"],\"BACKGROUND\": [\"...\"]})",
        # "v5": "Reply with a valid JSON containing the context sentence label for each context class (e.g: {\"INFORMATION\": [\"sent1\"],\"PERCEPTION\": [\"sent2\"],\"BACKGROUND\": [\"sent2\"]})",
        "v6": "Reply by surrounding context sequences wiht XML tags and adding a self closing XML tag containing the right function class directly after the #TARGET_REF  (e.g: <context>... #TARGET_REF<USE\> ... </context>)",
        "v7": "Reply with a valid JSON containing the context sequences and function labels (e.g: {\"label\": [\"USE\"], \"context\": [\"...\"]})",
        # "v8": "Reply with a valid JSON containing the context sentence labels and function labels (e.g: {\"label\": [\"USE\"], \"context\": [\"sent3\",\"sent4\"]})",
    },
    "further_inst": {
        #"v1": "Format all XML tags correctly, do not change the wording, do not exceed the input, do return anythig else that the input with added annotation"
    },
    "list_of_labels": {
        "v1":   """\n\nList Citation Function Classes\n
                BACKGROUND: Citation provides background Information. \n
                USE: Citation is used in citing paper. \n
                COMPARISON: Cited paper is judged as similar, different, or is agreed or disagreed with. \n
                MOTIVE: Citing paper is motivated by citation. \n
                EXTENSION: Citing paper extends citation.\n
                FUTURE: Citation mentions possible future work. \n
                """,
        "v2":   """\n\n\Citation Context Classes\n
                INFORMATION: context that describes information from the cited paper \n
                PERCEPTION: context that describes judgement or use of the cited paper \n
                BACKGROUND: context that is vaguely related to the citation (trends, figures, further information) \n
                """,
        "v3":   """\n\nList Citation Function Types\n
                BACKGROUND: Citation provides background Information. \n
                USE: Citation is used in citing paper. \n
                SIMILARITY: Citation is judged as similar, or is agreed with \n
                DIFFERENCE: Cited paper is judged as different, or is disagreed with.  \n
                MOTIVE: Citing paper is motivated by citation. \n
                EXTENDS: Citing paper extends citation. \n
                FUTURE: Citation mentions possible future work. \n
                """,
    },
    "reassure": {
        "v1": "Are the instructions clear to you?"
    },
    "reassurance": {
        "v1": "Yes, the instructions are clear to me. I will determine the right function class of the targeted citation (#TARGET_REF) based on the provided citation function classes and respond in XML fromat.",
        "v2": "Yes, the instructions are clear to me. I will determine the right function class of the targeted citation (#TARGET_REF) based on the provided citation function classes and respond in JSON fromat.",
        "v3": "Yes, the instructions are clear to me. I will annotate the citation context of the targeted citation (#TARGET_REF) based on the provided citation context classes and respond in XML fromat.",
        "v4": "Yes, the instructions are clear to me. I will annotate the citation context of the targeted citation (#TARGET_REF) based on the provided citation context classes and respond in JSON fromat.",
        "v5": "Yes, the instructions are clear to me. I will determine the right context and the function classes of the targeted citation (#TARGET_REF) based on the provided citation function classes and respond in XML fromat.",
        "v6": "Yes, the instructions are clear to me. I will determine the right context and the function classes of the targeted citation (#TARGET_REF) based on the provided citation function classes and respond in JSON fromat.",
    },
    "example_task": {
        "v1": "In a similar vain to #TARGET_REF and #REF , the method extends an existing flat shallow-parsing method to handle composite structures .",
        # "v2": "sent1: In a similar vain to #TARGET_REF and #REF , the method extends an existing flat shallow-parsing method to handle composite structures .",
    },
    "example_annotation":{
        "v1": "In a similar vain to #TARGET_REF<COMP/> and #REF , the method extends an existing flat shallow-parsing method to handle composite structures .",
        "v2": "{\"label\": [\"COMPARISON\"]}",
        "v3": "<PERCEPTION> In a similar vain to #TARGET_REF </PERCEPTION> and #REF , <BACKGROUND> the method extends an existing flat shallow-parsing method to handle composite structures </BACKGROUND> .",
        "v4": "{\"INFORMATION\": [], \"PERCEPTION\": [\"In a similar vain to #TARGET_REF\"], \"BACKGROUND\": [\"the method extends an existing flat shallow-parsing method to handle composite structures\"]}",
        # "v5": "{\"INFORMATION\": [], \"PERCEPTION\": [], \"BACKGROUND\": [\"sent1\"]}",
        "v6": "<CONT>In a similar vain to #TARGET_REF<SIMILARITY/><EXTENDS/> and #REF , the method extends an existing flat shallow-parsing method to handle composite structures </CONT> .",
        "v7": "{\"label\": [\"SIMILARITY\",\"EXTENDS\"], \"context\": [\"In a similar vain to #TARGET_REF and #REF , the method extends an existing flat shallow-parsing method to handle composite structures .\"]}",
        # "v8": "{\"label\": [\"SIMILARITIES\",\"EXTENDS\"], \"context\": [\"sent1\"]}",
    },
}

prompt_mapping = {
    "acl_arc": {
        "XML": [
            {"role": "user", "content": [("provided_material","v1"),("task_desc","v1"),('chain_of_thought','v1'),("output_desc","v1"),("list_of_labels","v1"),("reassure","v1")]},
            {"role": "assistant", "content": [('reassurance',"v1")]},
            {"role": "user", "content": [("example_task","v1")]},
            {"role": "assistant", "content": [("example_annotation","v1")]}
        ],
        "JSON": [
            {"role": "user", "content": [("provided_material","v1"),("task_desc","v1"),('chain_of_thought','v1'),("output_desc","v2"),("list_of_labels","v1"),("reassure","v1")]},
            {"role": "assistant", "content": [('reassurance',"v2")]},
            {"role": "user", "content": [("example_task","v1")]},
            {"role": "assistant", "content": [("example_annotation","v2")]}
        ],
        # "JSON2": [
        #     {"role": "user", "content": [("provided_material","v3"),("task_desc","v1"),("output_desc","v2"),("list_of_labels","v1"),("reassure","v1")]},
        #     {"role": "assistant", "content": [('reassurance',"v2")]},
        #     {"role": "user", "content": [("example_task","v2")]},
        #     {"role": "assistant", "content": [("example_annotation","v2")]}
        # ]
    },
    "finecite": {
        "XML": [
            {"role": "user", "content": [("provided_material","v2"),("task_desc","v2"),("output_desc","v3"),("list_of_labels","v2"),("reassure","v1")]},
            {"role": "assistant", "content": [('reassurance',"v3")]},
            {"role": "user", "content": [("example_task","v1")]},
            {"role": "assistant", "content": [("example_annotation","v3")]}
        ],
        "JSON": [
            {"role": "user", "content": [("provided_material","v2"),("task_desc","v2"),("output_desc","v4"),("list_of_labels","v2"),("reassure","v1")]},
            {"role": "assistant", "content": [('reassurance',"v4")]},
            {"role": "user", "content": [("example_task","v1")]},
            {"role": "assistant", "content": [("example_annotation","v4")]}
        ],
        # "JSON2": [
        #     {"role": "user", "content": [("provided_material","v4"),("task_desc","v2"),("output_desc","v5"),("list_of_labels","v2"),("reassure","v1")]},
        #     {"role": "assistant", "content": [('reassurance',"v4")]},
        #     {"role": "user", "content": [("example_task","v2")]},
        #     {"role": "assistant", "content": [("example_annotation","v5")]}
        # ]
    },
    "multicite": {
        "XML": [
            {"role": "user", "content": [("provided_material","v1"),("task_desc","v3"),("output_desc","v6"),("list_of_labels","v3"),("reassure","v1")]},
            {"role": "assistant", "content": [('reassurance',"v5")]},
            {"role": "user", "content": [("example_task","v1")]},
            {"role": "assistant", "content": [("example_annotation","v6")]}
        ],
        "JSON": [
            {"role": "user", "content": [("provided_material","v1"),("task_desc","v3"),("output_desc","v7"),("list_of_labels","v3"),("reassure","v1")]},
            {"role": "assistant", "content": [('reassurance',"v6")]},
            {"role": "user", "content": [("example_task","v1")]},
            {"role": "assistant", "content": [("example_annotation","v7")]}
        ],
        # "JSON2": [
        #     {"role": "user", "content": [("provided_material","v3"),("task_desc","v3"),("output_desc","v8"),("list_of_labels","v3"),("reassure","v1")]},
        #     {"role": "assistant", "content": [('reassurance',"v6")]},
        #     {"role": "user", "content": [("example_task","v2")]},
        #     {"role": "assistant", "content": [("example_annotation","v8")]}
        # ]
    },
}

class PromptForAutoCCA:
    
    def __init__(
        self,
        tokenizer,
        model_id,
        task,
        schema,
    ):
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.task = task
        self.schema = schema
        
    def create_sample(self, input_str, label_str, for_generation = False, return_prompt_only = False):
        prompt = self.get_prompt()
        if return_prompt_only: return prompt
        
        #add input_str
        prompt.append({"role": "user", "content": input_str})
        
        #add label_str if not for_generation
        if not for_generation: prompt.append({"role": "assistant", "content": label_str})
        
        #make simple prompt for sciTulu
        if self.model_id == 'scitulu':
            return
        else:
            encoded_input_ids = self.tokenizer.apply_chat_template(prompt)

        return {'input_ids': encoded_input_ids}

    def get_prompt(self):
        mappings = prompt_mapping[self.task][self.schema]
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