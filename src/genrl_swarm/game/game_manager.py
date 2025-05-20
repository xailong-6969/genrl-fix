import abc

from typing import Any, List

from genrl_swarm.state import GameState, GameNode
from genrl_swarm.rewards import RewardManager
from genrl_swarm.trainer import TrainerModule
from genrl_swarm.data import DataManager
from genrl_swarm.roles import RoleManager #TODO: Implement RoleManager+Pass to game manager

class GameManager(abc.ABC): #TODO: Make this use enum
    def __init__(self, 
                 game_state: GameState, 
                 reward_manager: RewardManager, 
                 trainer: TrainerModule, 
                 data_manager: DataManager, 
                 role_manager: RoleManager | None = None,
                 run_mode: str = "Train"
                 ):
        """Initialization method that stores the various managers needed to orchestrate this game"""
        self.state = game_state
        self.rewards = reward_manager
        self.trainer = trainer
        self.data_manager = data_manager
        self.roles = role_manager
        self.mode = run_mode.lower()
    
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
        #TODO(discuss): Need to explain this better... How do we make it more intuitive?
        pass

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

    #Core (default) game orchestration methods
    def run_game_stage(self):
        inputs = self.state.get_latest_state() # Fetches the current world state for all agents
        outputs = self.trainer.generate(inputs) # Generates a rollout. Ingests inputs indexable in the following way [Agent][Batch Item][Nodes idx within current stage][World state] then outputs something indexable as [Agent][Batch Item][Nodes idx within current stage]
        self.state.append_generation(outputs) # Adds the freshly generated rollout to the game state associated with this agent's nodes at this stage

    def run_game_round(self):
        # Loop through stages until end of round is hit
        while not self.end_of_round():
            self.run_game_stage() # Generates rollout and updates the game state
            world_states = self.communication.all_gather(self.state) #TODO(jari): Leaving as a placeholder for now
            self.state.advance_stage(world_states) # Prepare for next stage
        self.rewards.update_rewards(self.state) # Compute reward functions now that we have all the data needed for this round
        if self.mode in ['train', 'train_and_evaluate']:
            self.trainer.train(self.state, self.rewards)
        if self.mode in ['evaluate', 'train_and_evaluate']:
            self.trainer.evaluate(self.data_manager, self.rewards)
        self.state.advance_round(self.data_manager.get_round_data()) # Resets the game state appropriately, stages the next round, and increments round/stage counters appropriatelly

    def run_game(self):
        # Initialize game and/or run specific details of game state
        world_state_pruners = {"environment_pruner": self.environment_state_pruner, "opponent_pruner": self.opponent_state_pruner, "personal_pruner": self.personal_state_pruner}
        game_tree_brancher = {"terminal_node_decision_function": self.terminal_game_tree_node_decider, "stage_inheritance_function": self.stage_inheritance_function}
        self.state._init_game(self.data_manager.get_round_data(), world_state_pruners=world_state_pruners, game_tree_brancher=game_tree_brancher) # Prepare game trees within the game state for the initial round's batch of data
        
        # Loop through rounds until end of the game is hit
        while not self.end_of_game():
            self.run_game_round() # Loops through stages until end of round signal is received


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
                  "role_manager": role_manager, 
                  "run_mode": run_mode
                  }
        super().__init__(**kwargs)

    def end_of_game(self) -> bool:
        if self.state.round > self.max_round:
            return True
        else:
            return False
    
    def end_of_round(self) -> bool:
        if self.state.stage > self.max_stage:
            return True
        else:
            return False