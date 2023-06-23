from transformers import PreTrainedModel

def copy_model(model: PreTrainedModel) -> PreTrainedModel:
    """
    Copies a model.
    """
    model_copy = model.__class__.from_pretrained(model.config.pretrained_model_name_or_path)
    model_copy.load_state_dict(model.state_dict())
    return model_copy
