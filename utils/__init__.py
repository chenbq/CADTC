from .buffer import SuperReplayBuffer
from .buffer import ReplayBuffer
from .buffer import ReplayBufferATOC
from .buffer import ReplayBufferSched
from .buffer import ReplayBufferFAM
from .buffer import ReplayBufferAttention

REGISTRY = {}


REGISTRY["ATOC"] = ReplayBufferATOC
REGISTRY["ATOC_sim"] = ReplayBufferATOC
REGISTRY["FAM"] = ReplayBufferFAM
REGISTRY["Sched"] = ReplayBufferSched
REGISTRY["SchedO"] = ReplayBufferSched
REGISTRY["attention"] = ReplayBufferAttention