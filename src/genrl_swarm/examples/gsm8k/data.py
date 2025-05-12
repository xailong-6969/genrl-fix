import random
from datasets import load_dataset, Dataset
from genrl_swarm.data import DataManager

# --- Constants (System Prompts) ---
STAGE1_SYSTEM_PROMPT = """
You joined a mathematics study group. You are given a math question, and you want to come up with the best possible answer to share with the rest of the group. To ensure other understand your answer, first think through the reasoning needed to reach your final answer and then state your final answer.
An ideal answer will satisfy four important criteria: 1) The reasoning for your final answer will be in <think> </think> tags. 2) Your final answer to the question will be in <answer> </answer> tags. 3) Your reasoning will be correct, concise, and clearly related to the question. 4) The final answer you give will be the mathematically correct answer.
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

STAGE2_SYSTEM_PROMPT = """
You joined a mathematics study group. After being given a math question, all members of your study group have independantly come up with their own answer and you now want to decide which answer is best (or if no answer is correct). All students in the study group were instructed to give their reasoning process in <think> </think> tags and the final answer to the question in <answer> </answer> tags.
An ideal answer will satisfy four important criteria: 1) The reasoning for their final answer will be in <think> </think> tags. 2) Their final answer to the question will be in <answer> </answer> tags. 3) Their reasoning will be correct, concise, and clearly related to the question. 4) The final answer will be mathematically correct.
As a reminder, among all answers you have received, you want to decide which answer is best or if no answer is correct. You should compare the reasoning process of the different answers you've received, then explain why an answer is the best (or why no answer is correct), and finally you should state the unique student identifier (marked by <student> </student> tags) of the answer you believe is best or say "None" if no answer was correct.
Respond in the following format:
<compare>
...
</compare>
<explain>
...
</explain>
<identify>
...
</identify>
"""

STAGE3_SYSTEM_PROMPT = """
You joined a mathematics study group. After being given a math question, all members of your study group have independantly come up with their own answer and then compared all the proposed answers. You now have two tasks: 1) Consider the feedback/criticisms given by members of the study group and decide which answer you believe a majority of the group will agree is best (or say "None" if no answer was correct). 2) Incorporate details from the best answers, and the feedback/criticisms about these answers, to give the best possible answer to the question.
Before answering the question, all students in the study group were instructed to first give their reasoning process in <think> </think> tags and then give the final answer to the question in <answer> </answer> tags. Similarly, before comparing/criticizing the proposed answers, students in the study group were instructed to first compare the reasoning process of the different answers in <compare> </compare> tags and then to explain why an answer is best (or why no answer is correct) in <explain> </explain> tags and lastly to state the unique student identifier of the answer in <identify> </identify> tags.
After considering the feedback you must format your response as follows:
<summarize_feedback>
...
</summarize_feedback>
<majority>
...
</majority>
<question>
...
</question>
<think>
...
</think>
<answer>
...
</answer>
"""

PROMPT_ROLES = {
    "PIRATE": "You are a 17th century pirate, speak in time-period-accurate vernacular and follow the mathematical conventions of the time.",
    "PROFESSOR": "Your name is Professor Archibaldexposition and you are a mathematics professor at a prestigious university. Speak with eloquent and precise language.",
    "CHILD": "You are a 5 year old child who is very good at math. You sometimes make spelling errors or use improper grammar.",
    "ALIEN": "You are an alien from a distant galaxy who is trying to understand human mathematics. You sometimes confuse human customs or units of measure.",
    "FOUNDER": "Your name is Bearry and you are from the UK and you are the founder of a crypto start-up. Speak as you would during an investor meeting.",
}


class GSM8KDataManager(DataManager):
    def __init__(self,
                 num_students_to_sample: int | None = None,
                 num_critics_to_sample: int | None = None,
                 subsampling_method: str | None = None,
                 prompt_generator_role: str | None = None):
        super().__init__()

        self.STAGE1_SYSTEM_PROMPT = STAGE1_SYSTEM_PROMPT
        self.STAGE2_SYSTEM_PROMPT = STAGE2_SYSTEM_PROMPT
        self.STAGE3_SYSTEM_PROMPT = STAGE3_SYSTEM_PROMPT
        self.PROMPT_ROLES = PROMPT_ROLES

        self.num_students_to_sample = num_students_to_sample if num_students_to_sample is not None else 5
        self.num_critics_to_sample = num_critics_to_sample if num_critics_to_sample is not None else 5
        self.subsampling_method = subsampling_method if subsampling_method is not None else "top_k"
        self.prompt_generator_role = prompt_generator_role

    # --- Helper Methods ---
    def extract_hash_answer(self, text: str) -> str | None: 
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

    def generate_system_prompt(self, default_sys_prompt):
        if self.prompt_generator_role is None:
            return default_sys_prompt
        prompt_role_assignment = self.prompt_generator_role.upper()
        if prompt_role_assignment == "RANDOM":
            prompt_role_assignment = random.choice(list(self.PROMPT_ROLES.keys()))
        if prompt_role_assignment in self.PROMPT_ROLES:
            sys_prompt = self.PROMPT_ROLES[prompt_role_assignment] + default_sys_prompt
            return sys_prompt
        else:
            return default_sys_prompt

    def sorted_agent_ids(self, cols, prefix):
        student_ids = []
        for col_name in cols:
            if col_name.startswith(prefix):
                student_ids.append(col_name.split(prefix)[1])
        return sorted(student_ids, key=int)

    def get_unique_student_ids(self, cols):
        return {k: i for i, k in enumerate(self.sorted_agent_ids(cols, "agent_answers_"))}

    def get_unique_critic_ids(self, cols):
        return {k: i for i, k in enumerate(self.sorted_agent_ids(cols, "agent_opinion_"))}

    def pick_k_cols(self, cols, datum, current_stage, reward_fn=None):
        feature_prefix = "agent_answers_" if current_stage == 2 else "agent_opinion_"
        datum_cols = [c for c in cols if feature_prefix in c and c in datum]

        if not datum_cols:
            return []

        if current_stage == 2:
            default_k = self.num_students_to_sample
            method = self.subsampling_method
        elif current_stage == 3:
            default_k = self.num_critics_to_sample
            method = self.subsampling_method
        else: # Should ideally not be reached with current usage patterns
            default_k = min(len(datum_cols), 3) # Fallback default_k
            method = "random"

        if method == "random":
            return random.sample(datum_cols, min(len(datum_cols), default_k))
        
        elif method == "top_k":
            if reward_fn is None:
                return random.sample(datum_cols, min(len(datum_cols), default_k))

            completions_for_reward = [{feature_prefix[:-1]: datum[col]} for col in datum_cols] 
            current_prompt_list = datum.get("prompt")
            current_answer_str = datum.get("answer")

            if not current_prompt_list or current_answer_str is None:
                return random.sample(datum_cols, min(len(datum_cols), default_k))

            try:
                rewards = reward_fn(
                    prompts=[current_prompt_list],
                    completions=[completions_for_reward],
                    answer=[current_answer_str],
                )
            except Exception as e:
                # print(f"Error in reward_fn during pick_k_cols (stage {current_stage}): {e}. Defaulting to random.")
                return random.sample(datum_cols, min(len(datum_cols), default_k))
           
            if not isinstance(rewards, list) or (completions_for_reward and not rewards):
                # print(f"Invalid rewards format from reward_fn (stage {current_stage}). Defaulting to random.")
                return random.sample(datum_cols, min(len(datum_cols), default_k))

            if len(rewards) != len(datum_cols):
                # print(f"Reward count mismatch (stage {current_stage}). Defaulting to random.")
                return random.sample(datum_cols, min(len(datum_cols), default_k))
                
            sorted_features = [x for _, x in sorted(zip(rewards, datum_cols), key=lambda pair: pair[0], reverse=True)]
            return sorted_features[:default_k]
        return [] # Fallback for unknown method

    def generate_stage2_user_prompt(self, datum, cols, reward_fn=None):
        sp = []
        sp.append(f"The question we were given is: {datum['question']}" + "  \n\n")
        sp.append("The following answers to this question were suggested:" + " \n")
        picked_cols = self.pick_k_cols(cols, datum, current_stage=2, reward_fn=reward_fn)
        
        student_ids = self.get_unique_student_ids(picked_cols)
        for col in picked_cols:
            student_id = student_ids[col]
            sp.append(f"<student>Student #{student_id}</student> said \n{datum[col]}\n\n")
        return "".join(sp)

    def generate_stage3_user_prompt(self, datum, cols, reward_fn=None):
        sp = []
        sp.append(f"{datum['stage2_prompt_user_content']}" + "  \n")
        sp.append(
            "After comparing these answers, the following feedback was given about which answer is best:"
            + " \n"
        )
        picked_cols = self.pick_k_cols(cols, datum, current_stage=3, reward_fn=reward_fn)
        critic_ids = self.get_unique_critic_ids(picked_cols)
        for col in picked_cols:
            critic_id = critic_ids[col]
            sp.append(f"<criticism>Criticism #{critic_id}</criticism> was \n{datum[col]}\n\n")
        return "".join(sp)

    def get_gsm8k_questions(self, data) -> Dataset:
        sys_prompt = self.generate_system_prompt(self.STAGE1_SYSTEM_PROMPT)
        data = data.map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": x["question"]},
                ],
                "answer": self.extract_hash_answer(x["answer"]),
                "question_full": x["question"] 
            }
        )
        return data

    def get_gsm8k_questions_with_stage1_answers(self, data: Dataset, stage1_outputs_list_of_dicts, reward_fn=None) -> Dataset:
        stage1_outputs_dict = {item["question_full"]: item for item in stage1_outputs_list_of_dicts}
        sys_prompt = self.generate_system_prompt(self.STAGE2_SYSTEM_PROMPT)
        all_agent_answer_cols = list(self.get_unique_student_ids(stage1_outputs_list_of_dicts[0].keys() if stage1_outputs_list_of_dicts else []).keys())

        def map_fn(example):
            agent_answers_for_question = stage1_outputs_dict.get(example["question_full"], {})
            merged_example = {**example, **agent_answers_for_question}
            user_prompt_content = self.generate_stage2_user_prompt(merged_example, all_agent_answer_cols, reward_fn=reward_fn)
            return {
                "prompt": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt_content},
                ],
                "stage2_prompt_user_content": user_prompt_content
            }
        questions_with_answers = list(stage1_outputs_dict.keys())
        filtered_data = data.filter(lambda example: example["question_full"] in questions_with_answers)
        if not filtered_data and stage1_outputs_list_of_dicts:
            print("Warning: No matching questions found in the base dataset for the provided stage 1 outputs.")
        return filtered_data.map(map_fn) if filtered_data else Dataset.from_list([])

    def get_gsm8k_questions_with_stage1and2_answers(self, data: Dataset, stage2_outputs_list_of_dicts, reward_fn=None) -> Dataset:
        stage2_outputs_dict = {item["question_full"]: item for item in stage2_outputs_list_of_dicts}
        sys_prompt = self.generate_system_prompt(self.STAGE3_SYSTEM_PROMPT)
        all_agent_opinion_cols = list(self.get_unique_critic_ids(stage2_outputs_list_of_dicts[0].keys() if stage2_outputs_list_of_dicts else []).keys())

        def map_fn(example):
            agent_opinions_for_question = stage2_outputs_dict.get(example["question_full"], {})
            merged_example = {**example, **agent_opinions_for_question}
            user_prompt_content = self.generate_stage3_user_prompt(merged_example, all_agent_opinion_cols, reward_fn=reward_fn)
            return {
                "prompt": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt_content},
                ]
            }
        questions_with_opinions = list(stage2_outputs_dict.keys())
        filtered_data = data.filter(lambda example: example["question_full"] in questions_with_opinions)
        if not filtered_data and stage2_outputs_list_of_dicts:
            print("Warning: No matching questions found in the base dataset for the provided stage 2 outputs.")
        return filtered_data.map(map_fn) if filtered_data else Dataset.from_list([])

    def fill_unknown_answers_opinions(self, batch, cols_to_check, prefix):
        default_value_map = {
            "agent_answers_": "Unknown Answer",
            "agent_opinion_": "No Opinion"
        }
        default_value = default_value_map.get(prefix, "Unknown")
        num_examples = 0
        if batch:
            for key_batch in batch:
                if isinstance(batch[key_batch], list) and batch[key_batch]: 
                    num_examples = len(batch[key_batch])
                    break
            if num_examples == 0 and batch: 
                if any(isinstance(batch[key_batch_inner], list) for key_batch_inner in batch):
                    num_examples = 0 
        output_batch = {k: list(v) if isinstance(v, list) else v for k, v in batch.items()} 
        for col_name in cols_to_check:
            if col_name not in output_batch:
                output_batch[col_name] = [default_value] * num_examples
            else:
                current_col_values = output_batch[col_name]
                if not isinstance(current_col_values, list):
                    output_batch[col_name] = [default_value] * num_examples
                    continue
                new_col_values = [default_value if value is None else value for value in current_col_values]
                if len(new_col_values) < num_examples:
                    new_col_values.extend([default_value] * (num_examples - len(new_col_values)))
                output_batch[col_name] = new_col_values[:num_examples] 
        return output_batch

    # --- Main Data Methods (refactored from functions) ---
    def get_stage1_data(self):
        dataset_id = "openai/gsm8k"
        train_dataset_raw = load_dataset(dataset_id, "main", split="train")
        test_dataset_raw = load_dataset(dataset_id, "main", split="test")
        train_dataset_processed = self.get_gsm8k_questions(train_dataset_raw)
        test_dataset_processed = self.get_gsm8k_questions(test_dataset_raw)
        return train_dataset_processed, test_dataset_processed

    def get_stage2_data(self, stage1_outputs_list_of_dicts, stage1_reward_fn=None):
        if not stage1_outputs_list_of_dicts:
            print("Warning: stage1_outputs_list_of_dicts is empty. Returning an empty dataset for stage 2.")
            return Dataset.from_list([]) 
        base_data_for_stage2 = Dataset.from_list(stage1_outputs_list_of_dicts)
        all_possible_student_cols = list(self.get_unique_student_ids(stage1_outputs_list_of_dicts[0].keys()).keys())
        base_data_for_stage2 = base_data_for_stage2.map(
            self.fill_unknown_answers_opinions, 
            batched=True, 
            fn_kwargs={"cols_to_check": all_possible_student_cols, "prefix": "agent_answers_"}
        )
        stage2_dataset = self.get_gsm8k_questions_with_stage1_answers(base_data_for_stage2, stage1_outputs_list_of_dicts, reward_fn=stage1_reward_fn)
        return stage2_dataset

    def get_stage3_data(self, stage2_outputs_list_of_dicts, stage2_reward_fn=None):
        if not stage2_outputs_list_of_dicts:
            print("Warning: stage2_outputs_list_of_dicts is empty. Returning an empty dataset for stage 3.")
            return Dataset.from_list([])
        base_data_for_stage3 = Dataset.from_list(stage2_outputs_list_of_dicts)
        all_possible_critic_cols = list(self.get_unique_critic_ids(stage2_outputs_list_of_dicts[0].keys()).keys())
        base_data_for_stage3 = base_data_for_stage3.map(
            self.fill_unknown_answers_opinions, 
            batched=True, 
            fn_kwargs={"cols_to_check": all_possible_critic_cols, "prefix": "agent_opinion_"}
        )
        stage3_dataset = self.get_gsm8k_questions_with_stage1and2_answers(base_data_for_stage3, stage2_outputs_list_of_dicts, reward_fn=stage2_reward_fn)
        return stage3_dataset

    # --- GSM8KDataManager specific methods ---
    def initialize(self):
        # Placeholder for any specific initialization logic needed by the manager
        # For example, pre-loading datasets if they are static and large
        print(f"GSM8KDataManager initialized with: students_sample={self.num_students_to_sample}, critics_sample={self.num_critics_to_sample}, subsampling={self.subsampling_method}, role={self.prompt_generator_role}")
        pass
        
    def train_batch(self, stage: int = 1, previous_stage_outputs: list | None = None, reward_fn=None):
        if stage == 1:
            train_s1_data, _ = self.get_stage1_data()
            return train_s1_data
        elif stage == 2:
            if previous_stage_outputs is None:
                raise ValueError("previous_stage_outputs must be provided for stage 2 training data.")
            return self.get_stage2_data(previous_stage_outputs, stage1_reward_fn=reward_fn)
        elif stage == 3:
            if previous_stage_outputs is None:
                raise ValueError("previous_stage_outputs must be provided for stage 3 training data.")
            return self.get_stage3_data(previous_stage_outputs, stage2_reward_fn=reward_fn)
        else:
            raise ValueError(f"Unsupported stage for train_batch: {stage}")
        
    def eval_data(self, stage: int = 1, previous_stage_outputs: list | None = None, reward_fn=None, name: str | None = None):
        # 'name' could be 'train' or 'test' to select the appropriate split from stage 1
        # or used to identify specific evaluation sets later
        if stage == 1:
            # For stage 1, 'name' can differentiate between train/test splits for evaluation
            train_s1_data, test_s1_data = self.get_stage1_data()
            if name == "test":
                return test_s1_data
            return train_s1_data # Default to train for eval if not specified
        elif stage == 2:
            if previous_stage_outputs is None:
                raise ValueError("previous_stage_outputs must be provided for stage 2 eval data.")
            # Assuming eval also runs on outputs that would form the basis of the next training stage
            return self.get_stage2_data(previous_stage_outputs, stage1_reward_fn=reward_fn)
        elif stage == 3:
            if previous_stage_outputs is None:
                raise ValueError("previous_stage_outputs must be provided for stage 3 eval data.")
            return self.get_stage3_data(previous_stage_outputs, stage2_reward_fn=reward_fn)
        else:
            raise ValueError(f"Unsupported stage for eval_data: {stage}")