import os
from typing import Any, Dict, List, Tuple, Optional
from datasets import Dataset

from reasoning_gym.composite import CompositeConfig, CompositeDataset
from reasoning_gym.dataset import ReseedingDataset
from reasoning_gym.utils import SYSTEM_PROMPTS, compute_decimal_reward

from genrl_swarm.data import LocalMemoryTextDataManager


class ReasoningGymDataManager(LocalMemoryTextDataManager):
    """Data Manager for Reasoning Gym Datasets.
    
    This class integrates reasoning-gym's composite datasets with genrl-swarm's
    data management framework, providing infinite iteration through reseeding.
    """
    def __init__(
        self,
        yaml_config_path: str,
        num_train_samples: Optional[int] = None,
        num_evaluation_samples: Optional[int] = None,
        eval_split_ratio: float = 0.2,
        seed: Optional[int] = None,
        batch_item_id_column: Optional[str] = 'question',
        system_prompt_id: str = 'default',
        chunk_size: int = 500,
    ):
        """Initialize the ReasoningGymDataManager.
        
        Args:
            yaml_config_path: Path to the YAML configuration file for the composite dataset
            num_train_samples: Number of samples to use for training
            num_evaluation_samples: Number of samples to use for evaluation
            eval_split_ratio: Ratio of data to use for evaluation if num_evaluation_samples is None
            seed: Random seed for reproducibility
            batch_item_id_column: Column to use for batch item ID generation
            system_prompt_id: ID of system prompt from reasoning_gym.utils.SYSTEM_PROMPTS
            chunk_size: Size of chunks for ReseedingDataset
        """
        super().__init__(
            train_dataset=None,
            evaluation_dataset=None,
            num_train_samples=num_train_samples,
            num_evaluation_samples=num_evaluation_samples,
            column_name_map={'question': 'question', 'answer': 'answer', 'metadata': 'metadata'},
            column_preprocessing_map=None,
            seed=seed,
            batch_item_id_column=batch_item_id_column,
            data_generator=self.load_reasoning_gym_dataset  # TODO: this was confusing, we should document or change the way this is done
        )
        
        self.yaml_config_path = yaml_config_path
        self.eval_split_ratio = eval_split_ratio
        self.chunk_size = chunk_size
        self.system_prompt = SYSTEM_PROMPTS.get(system_prompt_id, SYSTEM_PROMPTS['default'])
        
        try:
            self.config = CompositeConfig.from_yaml(yaml_config_path)
            
            if seed is not None:
                self.config.seed = seed
                
            self.composite_dataset = CompositeDataset(self.config)
            
            self.reseeding_dataset = ReseedingDataset(self.composite_dataset, chunk_size=self.chunk_size)
            
            self._create_dataset_splits()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ReasoningGymDataManager: {str(e)}")
    
    def _create_dataset_splits(self):
        """Create train/eval dataset splits"""
        total_samples = len(self.composite_dataset)
        
        if self.num_samples['evaluation'] is None:
            eval_count = int(total_samples * self.eval_split_ratio)
        else:
            eval_count = min(self.num_samples['evaluation'], total_samples)
        
        if self.num_samples['train'] is None:
            train_count = total_samples - eval_count
        else:
            train_count = min(self.num_samples['train'], total_samples - eval_count)
        
        self.num_samples['train'] = train_count
        self.num_samples['evaluation'] = eval_count
        
        all_indices = list(range(total_samples))
        self.eval_indices = all_indices[-eval_count:] if eval_count > 0 else []
        self.train_indices = all_indices[:train_count]
    
    def load_reasoning_gym_dataset(
        self, 
        dataset_id_or_path: Optional[str] = None, 
        subset: Optional[str] = None,
        split: Optional[str] = 'train', 
        num_samples: Optional[int] = None
    ) -> Dataset:
        """Load the reasoning gym dataset from the composite dataset.
        
        This overrides the parent class's load_HF_dataset method.
        
        Args:
            dataset_id_or_path: Ignored, using composite dataset
            subset: Ignored, using composite dataset
            split: 'train' or 'test' to determine which split to use
            num_samples: Number of samples to use
            
        Returns:
            A Dataset object containing the samples from the composite dataset
        """
        dataset_dict = {'question': [], 'answer': [], 'metadata': []}
        
        if split in ('test', 'validation'):
            indices = self.eval_indices
            max_samples = self.num_samples['evaluation']
        else:  # Default to train
            indices = self.train_indices
            max_samples = self.num_samples['train']
        
        if num_samples is not None:
            max_samples = min(num_samples, max_samples)
        
        for i in range(max_samples):
            # Use modulo to handle cases where we need more samples than available
            idx = indices[i % len(indices)] if indices else i % len(self.composite_dataset)
            item = self.composite_dataset[idx]
            
            dataset_dict['question'].append(item['question'])
            dataset_dict['answer'].append(item['answer'])
            
            metadata = item.get('metadata', {})
            if not isinstance(metadata, dict):
                metadata = {'original_metadata': metadata}
            
            metadata['dataset_index'] = idx
            metadata['split'] = split
            
            dataset_dict['metadata'].append(metadata)
        
        return Dataset.from_dict(dataset_dict)
    
    # --- Helper Methods ---
    def state_to_system_prompt(self, state: Tuple[Any, Any, Any]) -> str:
        """Return the system prompt for the reasoning task."""
        return self.system_prompt
    
    def state_to_user_prompt(self, state: Tuple[Any, Any, Any]) -> str:
        """Convert the state to a user prompt."""
        return state[0]['question']
    
    def state_to_answer(self, state: Tuple[Any, Any, Any]) -> str:
        """Extract the answer from the state."""
        return state[0]['answer']
    
    def score_answer(self, predicted_answer: str, oracle_answer: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """Score an answer using the dataset's scoring function if available."""
        if metadata and 'source_dataset' in metadata:
            # Try to get the original dataset for scoring
            source_dataset = metadata['source_dataset']
            if source_dataset in self.composite_dataset.datasets:
                dataset = self.composite_dataset.datasets[source_dataset]
                # Create an entry with the oracle answer and metadata for scoring
                entry = {
                    'answer': oracle_answer,
                    'metadata': metadata  # Include the metadata in the entry
                }
                try:
                    return dataset.score_answer(predicted_answer, entry)
                except Exception as e:
                    print(f"Warning: Error using dataset scorer: {e}. Falling back to default scorer.")
        
        # Default to decimal reward computation from reasoning_gym.utils
        return compute_decimal_reward(predicted_answer, oracle_answer)
    
    # --- Required Methods ---
    def initialize(self):
        """Initialize the data manager."""
        print(f"Reasoning Gym Data Manager initialized with config: {self.yaml_config_path}")
        print(f"Loaded composite dataset with {len(self.composite_dataset)} samples")
        print(f"Train samples: {self.num_samples['train']}, Eval samples: {self.num_samples['evaluation']}")
        print(f"Dataset weights: {', '.join([f'{name}: {self.config.get_dataset_weight(name)}' for name in self.composite_dataset.datasets])}")
    
    def flatten_states(self, 
                      flattened_input: Dict[str, List[Any]], 
                      state: List[Any], 
                      stage: int
                      ) -> Dict[str, List[Any]]:
        """Convert the state into a flattened format for the model input."""
        if flattened_input == {}:
            flattened_input = {'system_prompt': [], 'user_prompt': [], 'answer': [], 'metadata': []}
        
        flattened_input['system_prompt'].append(self.state_to_system_prompt(state))
        flattened_input['user_prompt'].append(self.state_to_user_prompt(state))
        flattened_input['answer'].append(self.state_to_answer(state))
        
        if 'metadata' in state[0]:
            flattened_input['metadata'].append(state[0]['metadata'])
        else:
            flattened_input['metadata'].append({})
            
        return flattened_input
    
    def prepare_environment(self,
                           node_states: List[Any],
                           swarm_states: Dict[Any, Any],
                           stage: int,
                           agent: Any,
                           batch_id: Any
                           ) -> Any:
        """Prepare the environment state for the next stage."""
        return node_states[0]
    
    def prepare_opponent(self,
                        node_states: List[Any],
                        swarm_states: Dict[Any, Any],
                        stage: int,
                        agent: Any,
                        batch_id: Any
                        ) -> Any:
        """Prepare the opponent state for the next stage."""
        return self.filter_swarm_states(swarm_states=swarm_states, batch_id=batch_id)
    
    def prepare_personal(self,
                        node_states: List[Any],
                        swarm_states: Dict[Any, Any],
                        stage: int,
                        agent: Any,
                        batch_id: Any
                        ) -> Any:
        """Prepare the personal state for the next stage."""
        return None
