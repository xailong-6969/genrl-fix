import random
from datasets import load_dataset, Dataset
from typing import Dict, List, Tuple, Any, Callable
from torch import Tensor
from numpy import ndarray
import hashlib

from genrl_swarm.data import DataManager
from genrl_swarm.state import GameState

# --- Constants (System Prompts) ---
STAGE0_SYSTEM_PROMPT = """
You joined a mathematics study group. You are given a math problem, and you want to come up with the best possible answer to share with the rest of the group. Think through the solution of the problem step by step and then state your final answer.
An ideal solution will satisfy three important criteria: 1) Your step by step reasoning is correct, concise, and clearly related to the problem. 2) The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem. 3) The final answer you give will be the mathematically correct answer.
Remember to put your answer on its own line after \"Answer:\".
"""

STAGE1_SYSTEM_PROMPT = """
You are reviewing solutions to a given math problem that have been submitted by students in a study group. Your goal is to determine which solution is best amongst all the solutions you receive.
Before responding to the math problem all students in the study group were instructed to think through the solution of the problem step by step and then state their final answer on its own line after \"Answer:\".
Ideal solutions to the problem will satisfy three important criteria: 1) Their step by step reasoning is correct, concise, and clearly related to the problem. 2) The last line of the solution should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem. 3) The final answer is mathematically correct answer.
Give a step by step comparison of the different solutions you received and explain why a specific solution is the best according to the three stated criteria (or why no answer is correct).
The last line of your response should be of the form Choice: $Choice (without quotes) where $Choice is the unique student identifier (marked by <student> </student> tags) of the solution you believe was best or say "None" if no solution was correct.
Remember to put your final choice on its own line after \"Choice:\".
"""

STAGE2_SYSTEM_PROMPT = """
You are part of a mathematics study group. After receiving a math problem, all members of your study group independantly came up with their own solution and then compared all the proposed solution. Treat the best solutions to the problem and the feedback/criticisms about them as additional information, then think through the solution of the problem step by step again and state the final answer.
Before responding to the math problem all students in the study group were instructed to state their final answer on its own line after \"Answer:\". Similarly, before comparing/criticizing the proposed solutions, all students were instructed to put their final choice on its own line after \"Choice:\".
An ideal solution will satisfy three important criteria: 1) Your step by step reasoning is correct, concise, and clearly related to the problem. 2) The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem. 3) The final answer you give will be the mathematically correct answer.
Remember to put your answer on its own line after \"Answer:\".
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

        #TODO: Add in some attributed for users to specify the datasets they want to use+other data related params (e.g., seed and num_samples per batch)

        self.STAGE0_SYSTEM_PROMPT = STAGE0_SYSTEM_PROMPT
        self.STAGE1_SYSTEM_PROMPT = STAGE1_SYSTEM_PROMPT
        self.STAGE2_SYSTEM_PROMPT = STAGE2_SYSTEM_PROMPT
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

    def load_HF_dataset(self,
                        dataset_id: str = "/home/gensyn/shared/data/gsm8k",
                        split: str | None = 'train',
                        num_samples: int | None = None,
                        seed: int | None = None
                        ) -> Dataset:
        # Load dataset from HuggingFace
        dataset_raw = load_dataset(dataset_id, "main", split=split)
        if seed is not None:
            dataset_raw = dataset_raw.shuffle(seed=seed)
        if num_samples is not None:
            dataset_raw = dataset_raw.select(range(num_samples))
        return dataset_raw

    def generate_system_prompt(self, default_sys_prompt: str) -> str:
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
    
    # --- Main DataManager Methods ---
    def initialize(self):
        # NOTE: Placeholder for any specific initialization logic needed by the manager
        #       For example, pre-loading datasets if they are static and large
        print(f"GSM8KDataManager initialized with: students_sample={self.num_students_to_sample}, critics_sample={self.num_critics_to_sample}, subsampling={self.subsampling_method}, role={self.prompt_generator_role}")
        pass
        
    def get_round_data(self, 
                       dataset_id: str = "openai/gsm8k", 
                       split: str | None = 'train', 
                       num_samples: int | None = 10, 
                       seed: int | None = 561
                       ) -> List[Tuple[Any, Any, Any, Any]]:
        dataset_raw = self.load_HF_dataset(dataset_id=dataset_id, split=split, num_samples=num_samples, seed=seed)
        # Format it as world states + unique batch item identifier 
        # NOTE: Unique batch item id is only ever used for initializing game trees and for ensuring game states are only edited when communication is appropriate)
        dataset_processed = []
        for datum in dataset_raw:
            if 'gsm8k' in dataset_id.lower():
                env_state = {'question': datum['question'], 'answer': self.extract_hash_answer(datum['answer'])}
            elif ('dapo' in dataset_id.lower()) or ('big-math-rl' in dataset_id.lower()):
                env_state = {'question': datum['question'], 'answer': datum['solution']}
            else:
                env_state = {'question': datum['question'], 'answer': None}
            hash_fxn = hashlib.md5()
            hash_fxn.update(str.encode(env_state['question']))
            item = (int(hash_fxn.hexdigest(),16), env_state, None, None) #unique batch item id, environment_state, opponent_state, personal_state
            dataset_processed.append(item)
        return dataset_processed
     
    def get_eval_data(self, 
                      dataset_id: str = "openai/gsm8k", 
                      split: str | None = 'test', 
                      num_samples: int | None = None, 
                      seed: int | None = None
                      ) -> List[Tuple[Any, Any, Any]]:
        dataset_raw = self.load_HF_dataset(dataset_id=dataset_id, split=split, num_samples=num_samples, seed=seed)
        # Format it as world states
        dataset_processed = []
        for datum in dataset_raw:
            if 'gsm8k' in dataset_id.lower():
                env_state = {'question': datum['question'], 'answer': self.extract_hash_answer(datum['answer'])}
            elif ('dapo' in dataset_id.lower()) or ('big-math-rl' in dataset_id.lower()):
                env_state = {'question': datum['question'], 'answer': datum['solution']}
            else:
                env_state = {'question': datum['question'], 'answer': None}
            hash_fxn = hashlib.md5()
            hash_fxn.update(str.encode(env_state['question']))
            item = (int(hash_fxn.hexdigest(),16), env_state, None, None) #unique batch item id, environment_state, opponent_state, personal_state
            dataset_processed.append(item)
        return dataset_processed
    
    def prepare_input(self, 
                      inputs: Dict[Any, Dict[Any, List[Tuple[Any]]]], 
                      stage: int = None
                      ) -> Tuple[Dataset, Dict[int, Tuple[int, int, int]]]:
        if stage not in [0,1,2]:
            raise ValueError(f"Unsupported stage for prepare_input: {stage}")
        input_flattened, index_mapping = self.flatten_tree_input(inputs, stage)
        input_flattened = Dataset.from_dict(input_flattened)
        def data_map(datum):
            prompt = [{'role': 'system', 'content': datum['system_prompt']}, {'role': 'user', 'content': datum['user_prompt']}]
            answer = datum['answer']
            return {'prompt': prompt, 'answer': answer}
        input_prepared = input_flattened.map(data_map)
        return input_prepared, index_mapping
        
    def prepare_actions(self, outputs: Any, index_mapping: Dict[int, Tuple[Any]]) -> Dict[Any, List[List[Any]]]:
        if isinstance(outputs, Tensor | ndarray):
            outputs = outputs.tolist()
        actions = {}
        for idx, model_output in enumerate(outputs):
            agent, batch_id, node_idx = index_mapping[idx]
            if agent not in actions:
                actions[agent] = {}
            if batch_id not in actions[agent]:
                actions[agent][batch_id] = {}
            actions[agent][batch_id][node_idx] = model_output
        return actions

    def prepare_states(self, current_state: GameState, swarm_states: Dict[Any, Any]) -> Dict[Any, Dict[Any, List[Tuple[Any]]]]:
        # opponent_responses = self.filter_swarm_states(swarm_states) #NOTE: In multi-stage math this is nothing fancy and is appended equally on all nodes locally. Reduces to just ensuring the opponents' communication is a string or list of strings. In other games this may be more complex
        world_states = current_state.get_latest_state()
        for agent in world_states:
            for batch_id in world_states[agent]:
                for node_idx, state in enumerate(world_states[agent][batch_id]):
                    opponent_responses = self.filter_swarm_states(swarm_states, batch_id)
                    #Append the current stage's user prompt to the environment variable so that it persists into the next stage
                    world_states[agent][batch_id][node_idx][0]['prior_stage_user_prompt'] = self.state_to_user_prompt(state, current_state.stage)
                    #Replace opponent states with what was communicated by swarm
                    world_states[agent][batch_id][node_idx][1] = opponent_responses
                    #Set personal state to None by default
                    world_states[agent][batch_id][node_idx][2] = None
        
        return world_states

    # --- GSM8KDataManager specific methods ---
    def flatten_tree_input(self, 
                           inputs: Dict[Any, Dict[Any, List[Tuple[Any]]]], 
                           stage: int
                           ) -> Tuple[Dict[str, List[Any]], Dict[int, Tuple[int, int, int]]]: #TODO (Discuss): Is this or at least part of it worth providing as a more general util function? Not really GSM8K specific
        input_flattened, index_mapping = {'system_prompt': [], 'user_prompt': [], 'answer': []}, {}
        cur_idx = 0
        for agent in inputs:
            for batch_id in inputs[agent]:
                for node_idx, state in enumerate(inputs[agent][batch_id]):
                    input_flattened['system_prompt'].append(self.state_to_system_prompt(stage))
                    input_flattened['user_prompt'].append(self.state_to_user_prompt(state, stage))
                    input_flattened['answer'].append(self.state_to_answer(state))
                    
                    index_mapping[cur_idx] = (agent, batch_id, node_idx)
                    cur_idx += 1        
        return input_flattened, index_mapping
    
    def state_to_system_prompt(self, stage: int) -> str:
        if stage == 0:
            return self.generate_system_prompt(self.STAGE0_SYSTEM_PROMPT)
        elif stage == 1:
            return self.generate_system_prompt(self.STAGE1_SYSTEM_PROMPT)
        else:
            return self.generate_system_prompt(self.STAGE2_SYSTEM_PROMPT)

    def state_to_user_prompt(self, state: Tuple[Any, Any, Any], stage: int) -> str:
        if stage == 0:
            return state[0]['question'] #User prompt is just the math question in this case
        else:
            return self.append_to_last_stage_prompt(state, stage)

    def state_to_answer(self, state: Tuple[Any, Any, Any]) -> str:
        return state[0]['answer']
    
    def append_to_last_stage_prompt(self, state: Tuple[Any, Any, Any], stage: int) -> str:
        sp = []
        if stage == 1:
            sp.append(f"The given math problem is: {state[0]['question']}" + "  \n\n")
            sp.append("The following solutions were suggested for this problem:" + " \n")
            for idx, opponent_response in enumerate(state[1]): #NOTE: Assumes opponent states are already being stored as a list of generated strings from the opponent
                sp.append(f"--> Student #{idx} said: {opponent_response}\n\n")
        elif stage == 2:
            sp.append(f"{state[0]['prior_stage_user_prompt']}" + "  \n")
            sp.append("After comparing these solutions, the following feedback was given about which answer is best:" + " \n")
            for idx, opponent_response in enumerate(state[1]): #NOTE: Assumes opponent states are already being stored as a list of generated strings from the opponent
                sp.append(f"--> Criticism #{idx} was: {opponent_response}\n\n")
        else:
            raise ValueError(f"Unsupported stage for append_to_last_stage_prompt: {stage}")
        return "".join(sp)
    
    def filter_swarm_states(self, swarm_states: Dict[Any, Any], batch_id: Any) -> List[str]:
        opponent_responses = []
        for agent_id in swarm_states:
            if batch_id in swarm_states[agent_id]:
                for node_id in swarm_states[agent_id][batch_id]:
                    agent_action = swarm_states[agent_id][batch_id][node_id]
                    if isinstance(agent_action, str):
                        opponent_responses.append(agent_action)
                    elif isinstance(agent_action, list):
                        for response in agent_action:
                            if isinstance(response, str):
                                opponent_responses.append(response)
        return opponent_responses