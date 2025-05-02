from genrl_swarm.runner.global_defs import get_logger


class TestGameManager:
    def __init__(self, msg: str):
        self.msg = msg

    def run_game(self):
        get_logger().info(f"Run game with message: {self.msg}")
