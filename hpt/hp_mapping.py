from typing import Dict, Callable
from hpt.hp_space import hp_space_word_level_lr, hp_space_word_level

MAPPING_NAME_TO_HP_SPACE: Dict[str, Callable] = {
    "space_word_level_lr": hp_space_word_level_lr,
    "space_word_level": hp_space_word_level
}
