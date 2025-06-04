import abc
import hashlib
from typing import Any, Dict, List, Tuple, Callable
from copy import deepcopy
from datasets import load_dataset, Dataset
from torch import Tensor
from numpy import ndarray

from genrl_swarm.state import GameState
from genrl_swarm.data import DataManager
from genrl_swarm.misc_utils.utils import generate_md5_hash_id

class LocalMemoryTextDataManager(DataManager):
    def __init__(self,
                 train_dataset: str | None,
                 evaluation_dataset: str | None = None,
                 num_train_samples: int | None = 5,
                 num_evaluation_samples: int | None = None,
                 column_name_map: Dict[str,str] = None,
                 column_preprocessing_map: Dict[str, Callable] | None = None,
                 seed: int | None = None,
                 batch_item_id_column: str | None = None,
                 data_generator: Callable | None = None,
                 **kwargs
                 ):
        super().__init__()

        self.datasets = {'train': train_dataset, 'evaluation': evaluation_dataset}
        self.num_samples = {'train': num_train_samples, 'evaluation':num_evaluation_samples}
        self.column_map = {'names': column_name_map, 'preprocessing': column_preprocessing_map}
        self.seed = seed
        self.batch_item_id_column = batch_item_id_column
        if (data_generator is None) and (not isinstance(train_dataset, str)):
            raise ValueError('Provided train dataset is not a string, but no data generating function was provided. Please provide an appropriate path/dataset ID for your desired training data OR provide a function for generating data at the start of a round.')
        self.data_generator = data_generator or self.load_HF_dataset
        #Optional properties
        self.data_subset = kwargs.get("subsets", None)

    # --- Helper Methods ---
    def load_HF_dataset(self,
                        dataset_id_or_path: str,
                        subset: str | None = None,
                        split: str | None = 'train',
                        num_samples: int | None = None
                        ) -> Dataset:
        # Load dataset from HuggingFace
        if subset is not None:
            dataset_raw = load_dataset(dataset_id_or_path, subset, split=split)
        else:
            dataset_raw = load_dataset(dataset_id_or_path, split=split)
        if self.seed is not None:
            dataset_raw = dataset_raw.shuffle(seed=self.seed)
        if num_samples is not None:
            dataset_raw = dataset_raw.select(range(num_samples))
        return dataset_raw
    
    def filter_swarm_states(self, swarm_states: Dict[Any, Any], batch_id: Any) -> List[str]:
        '''
        Consumes data received from the communication step and unpacking it into something that will be combined with prior world-state to form a world-state for the next stage
        '''
        opponent_responses = []
        for agent_id in swarm_states:
            if batch_id in swarm_states[agent_id]:
                for node_idx, _ in enumerate(swarm_states[agent_id][batch_id]):
                    agent_action = swarm_states[agent_id][batch_id][node_idx]
                    if isinstance(agent_action, str):
                        opponent_responses.append(agent_action)
                    elif isinstance(agent_action, list):
                        for response in agent_action:
                            if isinstance(response, str):
                                opponent_responses.append(response)
        return opponent_responses
    
    def flatten_tree_input(self, 
                           inputs: Dict[Any, Dict[Any, List[Tuple[Any]]]], 
                           stage: int
                           ) -> Tuple[Dict[str, List[Any]], Dict[int, Tuple[int, int, int]]]:
        input_flattened, index_mapping = {}, {}
        cur_idx = 0
        for agent in inputs:
            for batch_id in inputs[agent]:
                for node_idx, state in enumerate(inputs[agent][batch_id]):
                    input_flattened = self.flatten_states(input_flattened, state, stage)                    
                    index_mapping[cur_idx] = (agent, batch_id, node_idx)
                    cur_idx += 1        
        return input_flattened, index_mapping
    
    # --- Optional Methods ---
    #NOTE: These require different approaches depending on the game being played/usecase, so should be hijacked when appropriate!
    def batch_item_id_generator(self, hashable_obj: Any) -> Any:
        '''
        Generates unique hashes for a given batch item. 
        '''
        return generate_md5_hash_id(hashable_obj)
    
    def prompt_map(self, flattened_data: Any) -> Any: #TODO: Come up with a better term than "flattened" data
        '''
        Maps flattened data into a prompt that will be consumed by the LLM
        '''
        prompt = [{'role': 'system', 'content': flattened_data['system_prompt']}, {'role': 'user', 'content': flattened_data['user_prompt']}]
        return {'prompt': prompt}
    
    def merge_swarm_and_node_states(self, 
                                    node_state: List[Any], 
                                    swarm_states: Dict[Any, Any], 
                                    stage: int, 
                                    agent: Any,
                                    batch_id: Any
                                    ) -> Tuple[Any, Any, Any]:
        '''
        Parses states from a node of in game tree as well as data coming from communication, and merges them into states that will be appended to nodes in the upcoming stage
        '''
        environment_state = self.prepare_environment(node_state, swarm_states, stage, agent, batch_id)
        environment_state['prior_stage_input_states'] = deepcopy(node_state)
        opponent_state = self.prepare_opponent(node_state, swarm_states, stage, agent, batch_id)
        personal_state = self.prepare_personal(node_state, swarm_states, stage, agent, batch_id)
        return environment_state, opponent_state, personal_state

    # --- Main DataManager Methods ---
    def get_round_data(self, **kwargs) -> List[Tuple[Any, Any, Any, Any]]:
        dataset_raw = self.data_generator(dataset_id_or_path=self.datasets['train'], subset=self.data_subset, split=kwargs.get("split", "train"), num_samples=self.num_samples['train'])
        dataset_processed = []
        for idx, datum in enumerate(dataset_raw): #TODO (gab): Split this into functions that return all three states and set this as default behaviour if they choose not to hijack it
            #Fill environment state with input data about the start of the round.
            if self.column_map['names'] is not None:
                env_state = {key: datum[self.column_map['names'][key]] for key in self.column_map['names']}
            elif getattr(dataset_raw, "column_names", False):
                env_state = {key: datum[key] for key in dataset_raw.column_names}
            else:
                raise AttributeError("No mapping for column names were provided and generated dataset object doesn't have a \"column_names\" method for inferring desired column names.")
            #Preprocess any columns if desired
            if self.column_map['preprocessing'] is not None:
                for col in self.column_map['preprocessing']:
                    if col in env_state:
                        env_state[col] = self.column_map['preprocessing'][col](env_state[col])
                    else:
                        raise ValueError(f"Received a column preprocessing function for column == {col}, but this column doesn't exist in your environment states whose columns are: {env_state.keys()}")
            if self.batch_item_id_column is not None:
                item = (self.batch_item_id_generator(env_state[self.batch_item_id_column]), env_state, None, None) #unique batch item id, environment_state, opponent_state, personal_state
            else:
                item = (idx, env_state, None, None) #unique batch item id, environment_state, opponent_state, personal_state
            dataset_processed.append(item)
        return dataset_processed
    
    def get_eval_data(self, **kwargs) -> List[Tuple[Any, Any, Any, Any]]:
        dataset_raw = self.data_generator(dataset_id_or_path=self.datasets['evaluation'], subset=self.data_subset, split=kwargs.get("split", "test"), num_samples=self.num_samples['evaluation'])
        dataset_processed = []
        for idx, datum in enumerate(dataset_raw): #TODO (gab): Split this into functions that return all three states and set this as default behaviour if they choose not to hijack it
            #Fill environment state with input data about the start of the round.
            if self.column_map['names'] is not None:
                env_state = {key: datum[self.column_map['names'][key]] for key in self.column_map['names']}
            elif getattr(dataset_raw, "column_names", False):
                env_state = {key: datum[key] for key in dataset_raw.column_names}
            else:
                raise AttributeError("No mapping for column names were provided and generated dataset object doesn't have a \"column_names\" method for inferring desired column names.")
            #Preprocess any columns if desired
            if self.column_map['preprocessing'] is not None:
                for col in self.column_map['preprocessing']:
                    if col in env_state:
                        env_state[col] = self.column_map['preprocessing'][col](env_state[col])
                    else:
                        raise ValueError(f"Received a column preprocessing function for column == {col}, but this column doesn't exist in your environment states whose columns are: {env_state.keys()}")
            if self.batch_item_id_column is not None:
                item = (self.batch_item_id_generator(env_state[self.batch_item_id_column]), env_state, None, None) #unique batch item id, environment_state, opponent_state, personal_state
            else:
                item = (idx, env_state, None, None) #unique batch item id, environment_state, opponent_state, personal_state
            dataset_processed.append(item)
        return dataset_processed
    
    def prepare_input(self, 
                      inputs: Dict[Any, Dict[Any, List[Tuple[Any]]]], 
                      stage: int = None
                      ) -> Tuple[Dataset, Dict[int, Tuple[int, int, int]]]:
        input_flattened, index_mapping = self.flatten_tree_input(inputs, stage)
        input_flattened = Dataset.from_dict(input_flattened)
        input_prepared = input_flattened.map(self.prompt_map)
        return input_prepared, index_mapping
    
    def prepare_actions(self, 
                        outputs: Any, 
                        index_mapping: Dict[int, Tuple[Any]]
                        ) -> Dict[Any, List[List[Any]]]:
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
        world_states = current_state.get_latest_state()
        for agent in world_states:
            for batch_id in world_states[agent]:
                for node_idx, node_state in enumerate(world_states[agent][batch_id]):
                    world_states[agent][batch_id][node_idx][0], world_states[agent][batch_id][node_idx][1], world_states[agent][batch_id][node_idx][2] = self.merge_swarm_and_node_states(node_state, swarm_states, current_state.stage, agent, batch_id)
        return world_states
    
    # --- Required Game-Dependant Methods ---
    @abc.abstractmethod
    def flatten_states(self, flattened_input: Dict[str, List[Any]], state: List[Any], stage: int) -> Dict[str, List[Any]]:
        """Return a dictionary keyed on columns for batched input to the model, where each key points to a ordered list of values each row of that column will have"""
        pass

    @abc.abstractmethod
    def prepare_environment(self,
                          node_states: List[Any],
                          swarm_states: Dict[Any, Any],
                          stage: int,
                          agent: Any,
                          batch_id: Any
                          ) -> Any:
        """
        Returns data that should be passed onto a node's children as an environment state when starting the next stage of the game. 
        NOTE: Said data can come from a node's states at the current stage, states received from communication with the swarm, and/or any other sources you choose to provide
        """
        pass
    
    @abc.abstractmethod
    def prepare_opponent(self,
                       node_states: List[Any],
                       swarm_states: Dict[Any, Any],
                       stage: int,
                       agent: Any,
                       batch_id: Any
                       ) -> Any:
        """
        Returns data that should be passed onto a node's children as an opponent state when starting the next stage of the game. 
        NOTE: Said data can come from a node's states at the current stage, states received from communication with the swarm, and/or any other sources you choose to provide
        """
        pass
    
    @abc.abstractmethod
    def prepare_personal(self,
                       node_states: List[Any],
                       swarm_states: Dict[Any, Any],
                       stage: int,
                       agent: Any,
                       batch_id: Any
                       ) -> Any:
        """
        Returns data that should be passed onto a node's children as an personal state when starting the next stage of the game. 
        NOTE: Said data can come from a node's states at the current stage, states received from communication with the swarm, and/or any other sources you choose to provide
        """
        pass