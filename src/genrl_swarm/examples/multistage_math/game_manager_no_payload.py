import hashlib
from typing import Any, List
from genrl_swarm.game import BaseGameManager, RunType
from random import sample

def generate_md5_hash_id(hashable_obj: Any) -> int: #TODO: Add this into a package-wide util during spring cleaning and remove from text data manager utils...
    hash_fxn = hashlib.md5()
    hash_fxn.update(str.encode(hashable_obj))
    return int(hash_fxn.hexdigest(),16)

class MSMGameManager(BaseGameManager):
    """
    Hijacks the Default GameManager just to keep it to a state from before adding in communication payload dataclasses.
    """
    def run_game_round(self):
        # Loop through stages until end of round is hit
        while not self.end_of_round():
            self.run_game_stage() # Generates rollout and updates the game state
            swarm_states = self.communication.all_gather_object(self.state.get_latest_actions()[self.rank])
            world_states = self.data_manager.prepare_states(self.state, swarm_states) #Maps states received via communication with the swarm to RL game tree world states
            self.state.advance_stage(world_states) # Prepare for next stage
    
        self.rewards.update_rewards(self.state) # Compute reward functions now that we have all the data needed for this round
        self._hook_after_rewards_updated() # Call hook

        if self.mode in [RunType.Train, RunType.TrainAndEvaluate]:
            self.trainer.train(self.state, self.data_manager, self.rewards) 
        if self.mode in [RunType.Evaluate, RunType.TrainAndEvaluate]:
            self.trainer.evaluate(self.state, self.data_manager, self.rewards)
    
        self.state.advance_round(self.data_manager.get_round_data(), agent_keys=self.agent_ids) # Resets the game state appropriately, stages the next round, and increments round/stage counters appropriatelly
        self.rewards.reset()
        self._hook_after_round_advanced() # Call hook

    def opponent_state_pruner(self, input: List[str]) -> Any:
        """
        Does deterministic random-k pruning on opponent states
        """
        hashed_inputs = [generate_md5_hash_id(i) for i in input]
        deterministic_scrambled_inputs = [i for _, i in sorted(zip(hashed_inputs, input))]
        return deterministic_scrambled_inputs[:self.prune_K]