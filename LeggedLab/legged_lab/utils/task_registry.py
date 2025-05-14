from typing import Tuple
from rsl_rl.env import VecEnv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnvConfig
    from legged_lab.envs.base.base_env_config import BaseAgentConfig


class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name: str, task_class: VecEnv, env_cfg: "BaseEnvConfig", train_cfg: "BaseAgentConfig"):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]

    def get_cfgs(self, name) -> Tuple["BaseEnvConfig", "BaseAgentConfig"]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        return env_cfg, train_cfg


# make global task registry
task_registry = TaskRegistry()
