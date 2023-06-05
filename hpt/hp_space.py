from typing import Any, Dict
import optuna


def hp_space_word_level_lr(trial: optuna.Trial) -> Dict[str, Any]:
    hp_space = {
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-5, 5e-5, 1e-4, 5e-4]),
    }
    return hp_space


def hp_space_word_level(trial: optuna.Trial) -> Dict[str, Any]:
    hp_space = {
        "temperature": trial.suggest_categorical("temperature", [1, 2, 5, 10]),
        "ce_alpha": trial.suggest_float("ce_alpha", 0,1, 0.9)
    }
    return hp_space
