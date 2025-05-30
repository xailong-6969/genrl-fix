import abc
from enum import Enum
from typing import Any, List, Tuple, Dict, Callable

from genrl_swarm.logging_utils.global_defs import get_logger
from genrl_swarm.state import GameState, GameNode
from genrl_swarm.rewards import RewardManager
from genrl_swarm.trainer import TrainerModule
from genrl_swarm.data import DataManager
from genrl_swarm.communication.communication import Communication
from genrl_swarm.roles import RoleManager #TODO: Implement RoleManager+Pass to game manager
from genrl_swarm.communication import Communication
from genrl_swarm.blockchain import SwarmCoordinator
from genrl_swarm.communication.hivemind.name_utils import get_name_from_peer_id


class RunType(Enum):
    Train = "train"
    Evaluate = "evaluate"
    TrainAndEvaluate = "train_and_evaluate"


class GameManager(abc.ABC): #TODO: Make this use enum
    def __init__(self, 
                 game_state: GameState, 
                 reward_manager: RewardManager, 
                 trainer: TrainerModule, 
                 data_manager: DataManager, 
                 communication: Communication | None = None,
                 role_manager: RoleManager | None = None,
                 run_mode: str = "train",
                 rank: int = 0,
                 **kwargs,
                 ):
        """Initialization method that stores the various managers needed to orchestrate this game"""
        self.state = game_state
        self.rewards = reward_manager
        self.trainer = trainer
        self.data_manager = data_manager
        self.communication = communication or Communication.create(**kwargs)
        self.roles = role_manager
        try:
            self.mode = RunType(run_mode)
        except ValueError:
            get_logger().info(f"Invalid run mode: {run_mode}. Defaulting to train only.")
            self.mode = RunType.Train
        self._rank = rank or self.communication.get_id()
        self.agent_ids = [self._rank] #NOTE: Add more if wanted for game/usecase

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

    def _hook_after_rewards_updated(self):
        """Hook method called after rewards are updated."""
        pass

    def _hook_after_round_advanced(self):
        """Hook method called after the round is advanced and rewards are reset."""
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
        self._hook_after_rewards_updated() # Call hook

        if self.mode in [RunType.Train, RunType.TrainAndEvaluate]:
            self.trainer.train(self.state, self.data_manager, self.rewards) #TODO(Discuss): Current implementation will treat all "local" agents the same and do a single training pass on one of them using data from all of them. Same with generation since trainer is single model-centric
        if self.mode in [RunType.Evaluate, RunType.TrainAndEvaluate]:
            self.trainer.evaluate(self.state, self.data_manager, self.rewards)
    
        self.state.advance_round(self.data_manager.get_round_data(), agent_keys=self.agent_ids) # Resets the game state appropriately, stages the next round, and increments round/stage counters appropriatelly
        self.rewards.reset()
        self._hook_after_round_advanced() # Call hook

    def run_game(self):
        # Initialize game and/or run specific details of game state
        world_state_pruners, game_tree_brancher = self.aggregate_game_state_methods()
        self.state._init_game(self.data_manager.get_round_data(), agent_keys=self.agent_ids, world_state_pruners=world_state_pruners, game_tree_brancher=game_tree_brancher) # Prepare game trees within the game state for the initial round's batch of data
        # Loop through rounds until end of the game is hit
        while not self.end_of_game():
            try:
                get_logger().info(f"Starting round: {self.state.round}/{getattr(self, 'max_round', None)}.")
                self.run_game_round() # Loops through stages until end of round signal is received
            except:
                self.trainer.cleanup()
                get_logger().exception("Exception occurred during game run.", stack_info=True)
                raise        


class DefaultGameManagerMixin:
    """
    Defines some default behaviour for games with a "shared memory", "linked list" game tree structure, and fixed duration, i.e. the next stage only ever has a single child and all state information from last stage can be "safely" inherited and nodes stop having children at a specific stage.
    """
    #Optional methods
    def environment_state_pruner(self, input: Any) -> Any:
        """
        Optional pruning function for environment states. The format and data types of environment states is game-specific, so exact behaviours should reflect this.
        WARNING: Output of this function is directly set as the environment state of nodes in game tree, which may in turn used for constructing input to your models/agents!
        """
        return input
    
    def opponent_state_pruner(self, input: Any) -> Any:
        """
        Optional pruning function for opponent states. The format and data types of opponent states is game-specific, so exact behaviours should reflect this.
        WARNING: Output of this function is directly set as the opponent state of nodes in game tree, which may in turn used for constructing input to your models/agents!
        """
        return input[:self.prune_K] #TODO: Come back and make this a top-k filter
    
    def personal_state_pruner(self, input: Any) -> Any:
        """
        Optional pruning function for personal states. The format and data types of personal states is game-specific, so exact behaviours should reflect this.
        WARNING: Output of this function is directly set as the personal state of nodes in game tree, which may in turn used for constructing input to your models/agents!
        """
        return input
    
    def terminal_game_tree_node_decider(self, stage_nodes: List[GameNode]) -> List[GameNode]:
        """
        Optional function defining whether the set of nodes from a stage are terminal and, hence, should not be branched.
        Input:
            List[GameNode]: List nodes from a stage in the game.
        Return:
            List[GameNode]: List of nodes from the game tree that should be designated as terminal/leaves. Empty list indicates no nodes should be set as terminal for this tree after said stage.
        """
        terminal = []
        for node in stage_nodes:
            if node["stage"] < self.max_stage-1: #NOTE: For custom terminal functions, you may want to add in your own more complex logic in here for deciding whether a node is terminal. For example, in something like chess, you may want to add logic for checking for checkmates, etc.
                pass
            else:
                terminal.append(node)
        return terminal

    def stage_inheritance_function(self, stage_nodes: List[GameNode]) -> List[List[GameNode]]:
        """
        Optional function defining whether the set of nodes from a stage are terminal and, hence, should not be branched.
        Input:
            List[GameNode]: List nodes from a stage in the game.
        Return:
            List[List[GameNode]]: List of lists of game nodes, where outer-most list contains a list for each node in the input (i.e., stage_nodes) and each inner-list contains children nodes.  
        """
        stage_children = []
        for i, node in enumerate(stage_nodes):
            children = []
            if not node._is_leaf_node(): #NOTE: For custom inheritance functions, you may want to add your own loop in here to generate several children according to whatever logic you desire
                child = GameNode(stage=node.stage+1,
                                 node_idx=0, #Will be overwritten by the game tree if not correct
                                 environment_states=node.environment_states,
                                 opponent_states=node.opponent_states,
                                 personal_states=node.personal_states,
                                 actions=None
                                 )
                children.append(child)
            stage_children.append(children)
        return stage_children
  

class BaseGameManager(DefaultGameManagerMixin, GameManager):
    """
    Basic GameManager with basic functionality baked-in.
    Will end the game when max_rounds is reached, end a round when max_stage is reached.
    """
    def __init__(self,
                 max_stage: int,
                 max_round: int,
                 prune_K: int,
                 game_state: GameState, 
                 reward_manager: RewardManager, 
                 trainer: TrainerModule, 
                 data_manager: DataManager, 
                 communication: Communication | None = None,
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


class SwarmGameManager(BaseGameManager):
    """GameManager that orchestrates a game using a SwarmCoordinator."""
    def __init__(self, 
                 coordinator: SwarmCoordinator, 
                 max_stage: int,
                 max_round: int,
                 prune_K: int,
                 game_state: GameState, 
                 reward_manager: RewardManager, 
                 trainer: TrainerModule, 
                 data_manager: DataManager, 
                 communication: Communication,
                 role_manager: RoleManager | None = None,
                 run_mode: str = "Train",
                 log_dir: str = "logs"
                 ):
        import logging 
        import os
        super().__init__(
            max_stage=max_stage,
            max_round=max_round,
            prune_K=prune_K,
            game_state=game_state,
            reward_manager=reward_manager,
            trainer=trainer,
            data_manager=data_manager,
            communication=communication,
            role_manager=role_manager,
            run_mode=run_mode
        )
        #Logging Setup
        peer_id = self.communication.get_id()
        animal_name = get_name_from_peer_id(peer_id)
        format_msg = f"[{animal_name}] %(asctime)s %(levelname)s: %(message)s"
        logging.basicConfig(level=logging.INFO, format=format_msg)
        formatter = logging.Formatter(format_msg)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"training_{animal_name}.log")
        )
        file_handler.setFormatter(formatter)
        _LOG = get_logger()
        _LOG.addHandler(file_handler)


        self.coordinator = coordinator
        round, stage = self.coordinator.get_round_and_stage()
        self.state.round = round
        self.state.stage = stage
        self.communication.step_ = self.state.round #initialize communication module to contract's round

        get_logger().info(f"🐱 Hello 🐈 [{animal_name}] 🦮 [{peer_id}]!")
        get_logger().info(f"bootnodes: {self.coordinator.get_bootnodes()}")
        get_logger().info(f"Starting round: {self.state.round}/{self.max_round}.")
        

    def _get_total_rewards(self):
        rewards = self.rewards.rewards
        if isinstance(rewards, list):
            if isinstance(rewards[0], list):
                flattened = [item for sublist in rewards for item in sublist]
                total_rewards = int(sum(flattened))
            else:
                print(rewards)
                total_rewards = int(sum(rewards))
        else:
            total_rewards = rewards
        return total_rewards

    def _hook_after_rewards_updated(self):
        # TODO: get rewards and submit
        # total_rewards = self._get_total_rewards()
        # self.coordinator.submit_reward(self.state.round, self.state.stage, int(total_rewards), self.communication.get_peer_id())
        pass
    
    def _hook_after_round_advanced(self):
        # TODO: get winners and submit
        # total_rewards = self._get_total_rewards()
        # winners = 'agent with max rewards I guess' # need to attribute rewards to peer_ids
        # self.coordinator.submit_winners(self.state.round, winners, self.communication.dht.peer_id)
        pass