from transformers.models.whisper import WhisperTokenizer, WhisperTokenizerFast
from datasets import DatasetDict

from dataloader.utils import get_map_function_to_restore_missing_special_tokens
from normalization.formatting import remove_casing_and_punctuation
from utils.distil_config import DistilConfig
from utils.constants import DEFAULT_TOKENIZER_MAX_LENGTH, DEFAULT_NUM_PROC


def postprocess_teacher_outputs(dataset_dict: DatasetDict,
                                config: DistilConfig) -> DatasetDict: 

    if config.distillation_num_beams == 1:
        tokenizer = WhisperTokenizerFast.from_pretrained(config.teacher_model_name_or_path, language=config.lang_name, task=config.task)

        dataset_dict = dataset_dict.map(lambda batch: {"teacher_text": tokenizer.batch_decode(batch["teacher_sequences"], skip_special_tokens=True)},
                                        batched=True)
        
        if config.postprocess_teacher:
            print("Remove casing and punctuation from the teacher's outputs...")
            dataset_dict = dataset_dict.map(lambda x: {"teacher_text": remove_casing_and_punctuation(x["teacher_text"])},
                                            num_proc=DEFAULT_NUM_PROC)
        
        if config.strip_teacher:
            print("Strip starting/ending whitespaces from the teacher's outputs...")
            dataset_dict = dataset_dict.map(lambda x: {"teacher_text": x["teacher_text"].strip()},
                                            num_proc=DEFAULT_NUM_PROC)
        
        dataset_dict = dataset_dict.map(lambda batch: {"teacher_sequences": tokenizer(batch["teacher_text"],
                                                                                      truncation=True,
                                                                                      max_length=DEFAULT_TOKENIZER_MAX_LENGTH).input_ids},
                                        batched=True, remove_columns=["teacher_text"])
        map_function_to_restore_missing_special_tokens = get_map_function_to_restore_missing_special_tokens(col="teacher_sequences",
                                                                                                            pretrained_model_name_or_path=config.student_model_name_or_path,
                                                                                                            language=config.lang_name,
                                                                                                            task=config.task)
        dataset_dict = dataset_dict.map(map_function_to_restore_missing_special_tokens, num_proc=DEFAULT_NUM_PROC)

    else:
        tokenizer = WhisperTokenizer.from_pretrained(config.teacher_model_name_or_path, language=config.lang_name, task=config.task)

        dataset_dict = dataset_dict.map(lambda x: {"teacher_text": tokenizer.batch_decode(x["teacher_sequences"], skip_special_tokens=True)},
                                        batched=False)
        if config.postprocess_teacher:
            print("Remove casing and punctuation from the teacher's outputs...")
            dataset_dict = dataset_dict.map(lambda x: {"teacher_text": [remove_casing_and_punctuation(seq) for seq in x["teacher_text"]]},
                                            num_proc=DEFAULT_NUM_PROC)
        
        if config.strip_teacher:
            print("Strip starting/ending whitespaces from the teacher's outputs...")
            dataset_dict = dataset_dict.map(lambda x: {"teacher_text": [seq.strip() for seq in x["teacher_text"]]},
                                            num_proc=DEFAULT_NUM_PROC)
        
        dataset_dict = dataset_dict.map(lambda x: {"teacher_sequences": tokenizer(x["teacher_text"],
                                                                                  truncation=True,
                                                                                  max_length=DEFAULT_TOKENIZER_MAX_LENGTH).input_ids},
                                        batched=False, remove_columns=["teacher_text"])

    return dataset_dict
