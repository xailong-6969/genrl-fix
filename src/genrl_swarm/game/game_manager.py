import abc
import os
import pickle
from datetime import datetime

from typing import Any, List, Tuple, Dict, Callable

from genrl_swarm.logging_utils.global_defs import get_logger
from genrl_swarm.state import GameState, GameNode
from genrl_swarm.rewards import RewardManager
from genrl_swarm.trainer import TrainerModule
from genrl_swarm.data import DataManager
from genrl_swarm.communication.communication import Communication
from genrl_swarm.roles import RoleManager #TODO: Implement RoleManager+Pass to game manager
from genrl_swarm.communication import Communication

class GameManager(abc.ABC): #TODO: Make this use enum
    def __init__(self, 
                 game_state: GameState, 
                 reward_manager: RewardManager, 
                 trainer: TrainerModule, 
                 data_manager: DataManager, 
                 communication: Communication,
                 role_manager: RoleManager | None = None,
                 run_mode: str = "Train",
                 rank: int = 0,
                 ):
        """Initialization method that stores the various managers needed to orchestrate this game"""
        self.state = game_state
        self.rewards = reward_manager
        self.trainer = trainer
        self.data_manager = data_manager
        self.communication = communication
        self.roles = role_manager
        self.mode = run_mode.lower()
        self._rank = rank

    @property
    def rank(self) -> int:
        return self._rank
    
    @rank.setter
    def rank(self, rank: int) -> None:
        self._rank = rank

    @abc.abstractmethod
    def end_of_game(self) -> bool:
        """
        Defines conditions for the game to end and no more rounds/stage should begin. 
        Return True if conditions imply game should end, else False
        """
        pass

    @abc.abstractmethod
    def end_of_round(self) -> bool:
        """
        Defines conditions for end of a round AND no more stages/"turns" should being for this round AND the game state should be reset for stage 0 of your game. 
        Return True if conditions imply game should end and no new round/stage should begin, else False
        """
        pass

    
    #Helper methods
    def aggregate_game_state_methods(self) -> Tuple[Dict[str, Callable], Dict[str, Callable]]:
        world_state_pruners = {"environment_pruner": getattr(self, "environment_state_pruner", None),
                               "opponent_pruner": getattr(self, "opponent_state_pruner", None), 
                               "personal_pruner": getattr(self, "personal_state_pruner", None)
                               }
        game_tree_brancher = {"terminal_node_decision_function": getattr(self, "terminal_game_tree_node_decider", None), 
                              "stage_inheritance_function": getattr(self, "stage_inheritance_function", None)
                              }
        return world_state_pruners, game_tree_brancher

    #Core (default) game orchestration methods
    def run_game_stage(self):
        inputs = self.state.get_latest_state() # Fetches the current world state for all agents
        inputs, index_mapping = self.data_manager.prepare_input(inputs, self.state.stage) # Maps game tree states to model ingestable inputs
        outputs = self.trainer.generate(inputs) # Generates a rollout. Ingests inputs indexable in the following way [Agent][Batch Item][Nodes idx within current stage][World state] then outputs something indexable as [Agent][Batch Item][Nodes idx within current stage]
        actions = self.data_manager.prepare_actions(outputs, index_mapping) # Maps model outputs to RL game tree actions
        self.state.append_actions(actions) # Adds the freshly generated rollout to the game state associated with this agent's nodes at this stage

    def run_game_round(self):
        # Loop through stages until end of round is hit
        while not self.end_of_round():
            self.run_game_stage() # Generates rollout and updates the game state #TODO(Discuss): Ugly, but gets the job done?
            swarm_states = self.communication.all_gather_object(self.state.get_latest_actions()[self.rank])
            world_states = self.data_manager.prepare_states(self.state, swarm_states) #Maps states received via communication with the swarm to RL game tree world states
            self.state.advance_stage(world_states) # Prepare for next stage
        self.rewards.update_rewards(self.state) # Compute reward functions now that we have all the data needed for this round
        if self.mode in ['train', 'train_and_evaluate']:
            self.trainer.train(self.state, self.data_manager, self.rewards) #TODO(Discuss): Current implementation will treat all "local" agents the same and do a single training pass on one of them using data from all of them. Same with generation since trainer is single model-centric
        if self.mode in ['evaluate', 'train_and_evaluate']:
            self.trainer.evaluate(self.state, self.data_manager, self.rewards)
        self.state.advance_round(self.data_manager.get_round_data()) # Resets the game state appropriately, stages the next round, and increments round/stage counters appropriatelly
        self.rewards.reset()

    def run_game(self):
        # Initialize game and/or run specific details of game state
        world_state_pruners, game_tree_brancher = self.aggregate_game_state_methods()
        self.state._init_game(self.data_manager.get_round_data(), world_state_pruners=world_state_pruners, game_tree_brancher=game_tree_brancher) # Prepare game trees within the game state for the initial round's batch of data
        # Loop through rounds until end of the game is hit
        while not self.end_of_game():
            try:
                get_logger().info(f"Starting round: {self.state.round}/{getattr(self, 'max_round', None)}.")
                self.run_game_round() # Loops through stages until end of round signal is received
            except:
                self.trainer.cleanup()
                get_logger().exception("Exception occurred during game run.", stack_info=True)
                raise        


class DefaultGameManagerMixin: #TODO: Add basic functionality to these methods!
    #Optional methods
    def environment_state_pruner(self) -> Any:
        """
        Optional pruning function for environment states. The format and data types of environment states is game-specific, so exact behaviours should reflect this.
        WARNING: Output of this function is directly set as the environment state of nodes in game tree, which may in turn used for constructing input to your models/agents!
        """
        pass
    
    def opponent_state_pruner(self) -> Any:
        """
        Optional pruning function for opponent states. The format and data types of opponent states is game-specific, so exact behaviours should reflect this.
        WARNING: Output of this function is directly set as the opponent state of nodes in game tree, which may in turn used for constructing input to your models/agents!
        """
        pass
    
    def personal_state_pruner(self) -> Any:
        """
        Optional pruning function for personal states. The format and data types of personal states is game-specific, so exact behaviours should reflect this.
        WARNING: Output of this function is directly set as the personal state of nodes in game tree, which may in turn used for constructing input to your models/agents!
        """
        pass
    
    def terminal_game_tree_node_decider(self, stage_nodes: List[GameNode]) -> List[GameNode]:
        """
        Optional function defining whether the set of nodes from a stage are terminal and, hence, should not be branched.
        Input:
            List[GameNode]: List nodes from a stage in the game.
        Return:
            List[GameNode]: List of nodes from the game tree that should be designated as terminal/leaves. Empty list indicates no nodes should be set as terminal for this tree after said stage.
        """
        return []

    def stage_inheritance_function(self, stage_nodes: List[GameNode]) -> List[List[GameNode]]:
        """
        Optional function defining whether the set of nodes from a stage are terminal and, hence, should not be branched.
        Input:
            List[GameNode]: List nodes from a stage in the game.
        Return:
            List[List[GameNode]]: List of lists of game nodes, where outer-most list contains a list for each node in the input (i.e., stage_nodes) and each inner-list contains children nodes.  
        """
        #TODO(discuss): Need to explain this better... How do we make it more intuitive?
        pass
  

class BaseGameManager(GameManager):
    """
    Default GameManager with some basic functionality baked-in.
    Will end the game when max_rounds is reached, end a round when max_stage is reached, and prune according to top-k rewards from previous stage.
    """
    def __init__(self,
                 max_stage: int,
                 max_round: int,
                 prune_K: int,
                 game_state: GameState, 
                 reward_manager: RewardManager, 
                 trainer: TrainerModule, 
                 data_manager: DataManager, 
                 communication: Communication,
                 role_manager: RoleManager | None = None,
                 run_mode: str = "Train"
                 ):
        """Init a GameManager which ends the game when max_rounds is reached, ends stage when max_stage is reached, and prunes according to top-k rewards"""
        self.max_stage = max_stage
        self.max_round = max_round
        self.prune_K = prune_K
        kwargs = {"game_state": game_state, 
                  "reward_manager": reward_manager, 
                  "trainer": trainer, 
                  "data_manager": data_manager, 
                  "communication": communication,
                  "role_manager": role_manager, 
                  "run_mode": run_mode
                  }
        super().__init__(**kwargs)

    def end_of_game(self) -> bool:
        if self.state.round < self.max_round:
            return False
        else:
            return True
    
    def end_of_round(self) -> bool:
        if self.state.stage < self.max_stage:
            return False
        else:
            return True