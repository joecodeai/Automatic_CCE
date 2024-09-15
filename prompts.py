prompt_building_blocks = {
    "provided_material": {
        "v1": "You are given an excerpt from a scientific text in which citations are marked #REF and #TARGET_REF. Further you are given a list of citation function classes.",
        "v2": "You are given an excerpt from a scientific text in which citations are marked #REF and #TARGET_REF. Further you are given a list of citation context classes.",
        "v3": "You are given an list of sentences from a scientific text in which citations are marked #REF and #TARGET_REF. Further you are given a list of citation function classes.",
        "v4": "You are given an list of sentences from a scientific text in which citations are marked #REF and #TARGET_REF. Further you are given a list of citation context classes.",
    },
    "task_desc":{
        "v1": "Your task is to assign one of the function classes to the targeted #TARGET_REF.",
        "v2": "Your task is it to annotate the citation context of the #TARGET_REF with the context classes",
        "v3": "Your task is it to assign function classes and extract the citation context of the #TARGET_REF",
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
        "v5": "Reply with a valid JSON containing the context sentence label for each context class (e.g: {\"INFORMATION\": [\"sent1\"],\"PERCEPTION\": [\"sent2\"],\"BACKGROUND\": [\"sent2\"]})",
        "v6": "Reply by surrounding context sequences wiht XML tags and adding a self closing XML tag containing the right function class directly after the #TARGET_REF  (e.g: <context>... #TARGET_REF<USE\> ... </context>)",
        "v7": "Reply with a valid JSON containing the context sequences and function labels (e.g: {\"label\": [\"USE\"], \"context\": [\"...\"]})",
        "v8": "Reply with a valid JSON containing the context sentence labels and function labels (e.g: {\"label\": [\"USE\"], \"context\": [\"sent3\",\"sent4\"]})",
    },
    "further_inst": {
        "v1": "Format all XML tags correctly, do not change the wording, do not exceed the input, do return anythig else that the input with added annotation"
    },
    "list_of_labels": {
        "v1":   """\n\nList Citation Function Classes\n
                BACKGROUND: The cited paper provides relevant background information or is part of the body of literature. \n
                USE: The citing paper uses the methodology or tools created by the cited paper. \n
                COMPARE_CONTRAST: The citing paper expresses similarities to or or differences from, or disagrees with, the cited paper.\n
                MOTIVATION: The citing paper is directly motivated by the cited paper. \n
                EXTENSION: The citing paper extends the methods, tools, or data of the cited paper.\n
                FUTURE: The cited paper is a potential avenue for future work. \n
                """,
        "v2":   """\n\n\Citation Context Classes\n
                INFORMATION: includes the information that are cited from the cited article \n
                PERCEPTION: perception or use of the cited information in the citing paper \n
                BACKGROUND: backgound or further information on why the citation is included \n
                """,
        "v3":   """\n\nList Citation Function Types\n
                BACKGROUND: The target paper provides relevant information for this domain. \n
                USE: The source paper uses an idea, method, tool, etc. of the target paper \n
                SIMILARITIES: The source paper expresses similarities towards the target paper. Either similarities between the source and the target paper or similarities between another publication and the target paper. \n
                DIFFERENCES: The source paper expresses differences towards the target paper. Either differences between the source and the target paper or differences between another publication and the target paper. \n
                MOTIVATION: The target paper provides motivation for the source paper. For instance, it illustrates the need for data, goals, methods etc. \n
                EXTENDS: The source paper extends an idea, method, tool, etc. of the target paper. \n
                FUTURE: The target paper is a potential avenue for future research. Often corresponds to hedging or speculative language about work not yet performed. \n
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
        "v2": "sent1: In a similar vain to #TARGET_REF and #REF , the method extends an existing flat shallow-parsing method to handle composite structures .",
    },
    "example_annotation":{
        "v1": "In a similar vain to #TARGET_REF<COMPARE_CONTRAST/> and #REF , the method extends an existing flat shallow-parsing method to handle composite structures .",
        "v2": "{\"label\": [\"COMPARE_CONTRAST\"]}",
        "v3": "<PERCEPTION> In a similar vain to #TARGET_REF </PERCEPTION> and #REF , <BACKGROUND> the method extends an existing flat shallow-parsing method to handle composite structures </BACKGROUND> .",
        "v4": "{\"INFORMATION\": [], \"PERCEPTION\": [\"In a similar vain to #TARGET_REF\"], \"BACKGROUND\": [\"the method extends an existing flat shallow-parsing method to handle composite structures\"]}",
        "v5": "{\"INFORMATION\": [], \"PERCEPTION\": [], \"BACKGROUND\": [\"sent1\"]}",
        "v6": "<context>In a similar vain to #TARGET_REF<SIMILARITIES/><EXTENDS/> and #REF , the method extends an existing flat shallow-parsing method to handle composite structures </context> .",
        "v7": "{\"label\": [\"SIMILARITIES\",\"EXTENDS\"], \"context\": [\"In a similar vain to #TARGET_REF and #REF , the method extends an existing flat shallow-parsing method to handle composite structures .\"]}",
        "v8": "{\"label\": [\"SIMILARITIES\",\"EXTENDS\"], \"context\": [\"sent1\"]}",
    },
}

prompt_mapping = {
    "acl_arc": {
        "XML": [
            {"role": "user", "content": [("provided_material","v1"),("task_desc","v1"),("chain_of_thought","v1"),("output_desc","v1"),('further_inst',"v1"),("list_of_labels","v1"),("reassure","v1")]},
            {"role": "assistant", "content": [('reassurance',"v1")]},
            {"role": "user", "content": [("example_task","v1")]},
            {"role": "assistant", "content": [("example_annotation","v1")]}
        ],
        "JSON1": [
            {"role": "user", "content": [("provided_material","v1"),("task_desc","v1"),("chain_of_thought","v1"),("output_desc","v2"),("list_of_labels","v1"),("reassure","v1")]},
            {"role": "assistant", "content": [('reassurance',"v2")]},
            {"role": "user", "content": [("example_task","v1")]},
            {"role": "assistant", "content": [("example_annotation","v2")]}
        ],
        "JSON2": [
            {"role": "user", "content": [("provided_material","v3"),("task_desc","v1"),("chain_of_thought","v1"),("output_desc","v2"),("list_of_labels","v1"),("reassure","v1")]},
            {"role": "assistant", "content": [('reassurance',"v2")]},
            {"role": "user", "content": [("example_task","v2")]},
            {"role": "assistant", "content": [("example_annotation","v2")]}
        ]
    },
    "finecite": {
        "XML": [
            {"role": "user", "content": [("provided_material","v2"),("task_desc","v2"),("chain_of_thought","v2"),("output_desc","v3"),('further_inst',"v1"),("list_of_labels","v2"),("reassure","v1")]},
            {"role": "assistant", "content": [('reassurance',"v3")]},
            {"role": "user", "content": [("example_task","v1")]},
            {"role": "assistant", "content": [("example_annotation","v3")]}
        ],
        "JSON1": [
            {"role": "user", "content": [("provided_material","v2"),("task_desc","v2"),("chain_of_thought","v2"),("output_desc","v4"),("list_of_labels","v2"),("reassure","v1")]},
            {"role": "assistant", "content": [('reassurance',"v4")]},
            {"role": "user", "content": [("example_task","v1")]},
            {"role": "assistant", "content": [("example_annotation","v4")]}
        ],
        "JSON2": [
            {"role": "user", "content": [("provided_material","v4"),("task_desc","v2"),("chain_of_thought","v2"),("output_desc","v5"),("list_of_labels","v2"),("reassure","v1")]},
            {"role": "assistant", "content": [('reassurance',"v4")]},
            {"role": "user", "content": [("example_task","v2")]},
            {"role": "assistant", "content": [("example_annotation","v5")]}
        ]
    },
    "multicite": {
        "XML": [
            {"role": "user", "content": [("provided_material","v1"),("task_desc","v3"),("chain_of_thought","v3"),("output_desc","v6"),('further_inst',"v1"),("list_of_labels","v3"),("reassure","v1")]},
            {"role": "assistant", "content": [('reassurance',"v5")]},
            {"role": "user", "content": [("example_task","v1")]},
            {"role": "assistant", "content": [("example_annotation","v6")]}
        ],
        "JSON1": [
            {"role": "user", "content": [("provided_material","v1"),("task_desc","v3"),("chain_of_thought","v3"),("output_desc","v7"),("list_of_labels","v3"),("reassure","v1")]},
            {"role": "assistant", "content": [('reassurance',"v6")]},
            {"role": "user", "content": [("example_task","v1")]},
            {"role": "assistant", "content": [("example_annotation","v7")]}
        ],
        "JSON2": [
            {"role": "user", "content": [("provided_material","v3"),("task_desc","v3"),("chain_of_thought","v3"),("output_desc","v8"),("list_of_labels","v3"),("reassure","v1")]},
            {"role": "assistant", "content": [('reassurance',"v6")]},
            {"role": "user", "content": [("example_task","v2")]},
            {"role": "assistant", "content": [("example_annotation","v8")]}
        ]
    },
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
        
    def create_sample(self, input_str, label_str, for_generation = False, return_prompt_only = False):
        prompt = self.get_prompt()
        if return_prompt_only: return prompt
        
        #add input_str
        prompt.append({"role": "user", "content": input_str})
        
        #add label_str if not for_generation
        if not for_generation: prompt.append({"role": "assistant", "content": label_str})
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