from transformers import WhisperTokenizer
from utils.distil_config import DistilConfig


def assert_if_distillation_tokenizers_match(config: DistilConfig) -> None:
    """
    Sanity checks for distillation experiments.
    """
    
    # --- Tokenizer ---
    teacher_tokenizer = WhisperTokenizer.from_pretrained(
        config.student_model_name_or_path,
        language=config.lang_name,
        task=config.task
    )
    student_tokenizer = WhisperTokenizer.from_pretrained(
        config.student_model_name_or_path,
        language=config.lang_name,
        task=config.task
    )
    
    # Assert that teacher_processor and student_processor are the same at the exception
    # of the name_or_path attribute:
    list_attributes = ["vocab_size", "model_max_length", "is_fast", "padding_side",
                       "truncation_side", "clean_up_tokenization_spaces"]
    for attribute in list_attributes:
        assert getattr(teacher_tokenizer, attribute) == getattr(student_tokenizer, attribute), \
            f"The teacher and the student must have the same tokenizer.{attribute} attribute."
    
    return
