from transformers import WhisperForConditionalGeneration

class WhisperForConditionalGenerationForDistillation(WhisperForConditionalGeneration):
    # TODO: experimental -> https://github.com/huggingface/transformers/issues/20873#issuecomment-1362940576
    def prepare_inputs_for_generation(self, *args, added_param=None, **kwargs):
            output = super().prepare_inputs_for_generation(*args, **kwargs)
            if "teacher_sequences" in output:
                import pdb; pdb.set_trace()
            # (Pdb) output.keys()
            # dict_keys(['encoder_outputs', 'past_key_values', 'decoder_input_ids', 'use_cache', 'decoder_attention_mask'])
            # output.update({"added_param": added_param})
            return output
