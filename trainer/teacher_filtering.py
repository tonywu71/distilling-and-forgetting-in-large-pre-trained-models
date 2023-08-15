from datasets import DatasetDict
from dataloader.filtering import filter_samples_1_best
from k_beam_search.smart_load_k_beam_search import smart_load_dataset_with_k_beam_search

from utils.distil_config import DistilConfig
from utils.constants import DEFAULT_NUM_PROC


def filter_teacher_outputs(dataset_dict: DatasetDict,
                           config: DistilConfig) -> DatasetDict:
    is_seq_level = config.method_distil in ["seq_level_uniform", "seq_level_ranked"]
    
    if is_seq_level:
        print("Filtering out samples for which the teacher got poor transcriptions...")
        if config.distillation_num_beams == 1:
            dataset_dict["train"] = filter_samples_1_best(ds=dataset_dict["train"], config=config)
        else:
            for k in range(config.distillation_num_beams):
                dataset_dict["train"] = dataset_dict["train"].map(lambda x: {"teacher_sequences_copy": x["teacher_sequences"],
                                                                            "teacher_sequences": x["teacher_sequences"][k]},
                                                                    num_proc=DEFAULT_NUM_PROC)
                dataset_dict["train"] = filter_samples_1_best(ds=dataset_dict["train"], config=config)
                dataset_dict["train"] = dataset_dict["train"].map(lambda x: {"teacher_sequences": x["teacher_sequences_copy"]},
                                                                    num_proc=DEFAULT_NUM_PROC)
                dataset_dict["train"] = dataset_dict["train"].remove_columns("teacher_sequences_copy")
    
    else:  # word-level
        config.method_distil = "seq_level_uniform"  # hotfix for `smart_load_dataset_with_k_beam_search`
        config.distillation_num_beams = 1
        dataset_dict = smart_load_dataset_with_k_beam_search(config=config,
                                                             dataset_dict=dataset_dict)
        dataset_dict["train"] = filter_samples_1_best(ds=dataset_dict["train"], config=config)
        config.method_distil = "word_level"
        config.distillation_num_beams = None
    
    return dataset_dict
