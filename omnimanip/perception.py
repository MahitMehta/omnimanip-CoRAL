"""Task-relevant object grounding and stage partitioning (paper Sec. 3.1, Fig. 2)."""

from . import vlm


def partition(instruction, marked):
    """VLM filters task-relevant objects on the set-of-mark image and decomposes the task into
    stages S_i = {A_i, O_active, O_passive}. bbox_IDs are the mark ids; -1 = not marked."""
    return vlm.query("task_stage_partition", [marked], instruction=instruction)
