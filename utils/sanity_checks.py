from transformers import WhisperTokenizer
from utils.distil_config import DistilConfig


def distillation_sanity_check(config: DistilConfig) -> None:
    """
    Sanity checks for distillation experiments.
    """
    
    teacher_processor = WhisperTokenizer.from_pretrained(
        config.student_model_name_or_path,
        language=config.lang_name,
        task=config.task
    )
    student_processor = WhisperTokenizer.from_pretrained(
        config.student_model_name_or_path,
        language=config.lang_name,
        task=config.task
    )
    
    assert teacher_processor.tokenizer == student_processor.tokenizer, \
        "The teacher and the student must have the same tokenizer."
