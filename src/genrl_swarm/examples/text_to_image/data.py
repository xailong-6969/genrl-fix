from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Callable, Optional, Union
from torch.utils.data import DataLoader, Dataset
import numpy as np
from genrl_swarm.state import GameState
from genrl_swarm.data.data_manager import DataManager
from genrl_swarm.examples.text_to_image.ddpo_trainer import DDPOGeneratedOutput, DDPOSample

class RandomPromptDataset(Dataset):
    def __init__(self, path):
        with open(path) as f:
            dataset = f.readlines()
        self.dataset = [line.strip() for line in dataset]

    def __call__(self):
        return np.random.choice(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


def custom_collate_fn(batch):
    """Collates batches into (list_of_prompts, single_metadata_dict)."""
    prompts = [(item, None, None) for item in batch] # env state, opponent state, personal state
    return prompts

class LocalDatasetManager(DataManager):
    def __init__(self, train_dataset_path, eval_dataset_path, train_batch_size: int=10, num_eval_samples: int=1, sample_num_batches_per_round: int=1):
        train_dataset = RandomPromptDataset(train_dataset_path)
        eval_dataset = RandomPromptDataset(eval_dataset_path)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=custom_collate_fn  # Use custom collate function
        )
        self.eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=num_eval_samples,
            shuffle=False,
            drop_last=True,
            collate_fn=custom_collate_fn  # Use custom collate function
        )
        # self.sample_num_batches_per_round = sample_num_batches_per_round    
        self._train_iterator = None

    def train_batch(self):
        if self._train_iterator is None:
            self._train_iterator = iter(self.train_dataloader)

        try:
            return next(self._train_iterator)

        except StopIteration:
            self._train_iterator = iter(self.train_dataloader)
            return next(self._train_iterator)
    
    def get_round_data(self):
        return self.train_batch() # TODO: Add support for multiple batches per round

    def get_eval_data(self, name: str | None = None):
        return self.eval_dataloader

    def prepare_input(self, inputs: Dict[Any, List[List[Tuple[Any]]]], stage: int):
        batch = []
        index_mapping = {}
        idx = 0
        for agent in inputs:
            for batch_idx, batch_item in enumerate(inputs[agent]):
                for node_idx, node in enumerate(batch_item):
                    world_state = node[0]
                    opponent_state = node[1]
                    personal_state = node[2]
                    batch.append(world_state)
                    index_mapping[idx] = (agent, batch_idx, node_idx)
                    idx += 1

        return batch, index_mapping

    def prepare_actions(self, outputs: Any, index_mapping: Dict[int, Tuple[Any]]) -> Dict[Any, List[List[Any]]]:
        actions = defaultdict(list)
        samples = outputs.samples
        prompt_image_pairs = outputs.prompt_image_pairs
        prompts = prompt_image_pairs[0]
        images = prompt_image_pairs[1]
        for idx in range(samples.prompt_embeds.shape[0]):
            agent, batch_idx, node_idx = index_mapping[idx]
            if agent not in actions:
                actions[agent] = {}
            if batch_idx not in actions[agent]:
                actions[agent][batch_idx] = {}

            repackaged_sample = DDPOGeneratedOutput(
                samples=DDPOSample(
                prompt_embeds=samples.prompt_embeds[idx],
                timesteps=samples.timesteps[idx],
                latents=samples.latents[idx],
                next_latents=samples.next_latents[idx],
                log_probs=samples.log_probs[idx],
                negative_prompt_embeds=samples.negative_prompt_embeds[idx],
            ),
            prompt_image_pairs=[prompts[idx], images[idx]]
            )
            actions[agent][batch_idx][node_idx] = [repackaged_sample]
        return actions

    def prepare_states(self, current_state: GameState, swarm_states: Dict[Any, Any]) -> Dict[Any, List[List[Tuple[Any]]]]:
        world_states = current_state.get_latest_state()
        for agent in world_states:
            for batch_idx, batch in enumerate(world_states[agent]):
                for node_idx, node in enumerate(batch):
                    world_states[agent][batch_idx][node_idx][0] = swarm_states[agent][batch_idx][node_idx][0].prompt_image_pairs 
                    world_states[agent][batch_idx][node_idx][1] = swarm_states[agent][batch_idx][node_idx][0].samples 
                    world_states[agent][batch_idx][node_idx][2] = None
        return world_states

