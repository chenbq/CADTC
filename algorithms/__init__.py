from .ACML import ACML
from .ATOCNet import ATOCNet
from .ATOCNet_sim import ATOCNet_sim
from .maddpg import MADDPG
from .ATOCNetHalf import ATOCNetHalf

REGISTRY = {}

REGISTRY["ACML"] = ACML
REGISTRY["ATOC"] = ATOCNet_sim #ATOCNet #ATOCNetHalf #11-04 to test the impact of skip connection of atoc noSC
REGISTRY["ATOC_noSC"] = ATOCNetHalf
REGISTRY["ATOC_sim"] = ATOCNet_sim
REGISTRY["maddpg"] = MADDPG
REGISTRY["ddpg"] = MADDPG





