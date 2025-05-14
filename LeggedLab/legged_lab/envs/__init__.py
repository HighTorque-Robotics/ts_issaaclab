from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseEnvCfg, BaseAgentCfg
from legged_lab.envs.pai.pai_config import PaiFlatEnvCfg, PaiRoughEnvCfg, PaiFlatAgentCfg, PaiRoughAgentCfg
from legged_lab.utils.task_registry import task_registry

task_registry.register("pai_flat", BaseEnv, PaiFlatEnvCfg(), PaiFlatAgentCfg())
task_registry.register("pai_rough", BaseEnv, PaiRoughEnvCfg(), PaiRoughAgentCfg())
