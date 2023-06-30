from transformers.models.whisper import WhisperForConditionalGeneration

def get_ewc_params(model: WhisperForConditionalGeneration,
                   
                   ) -> dict[str, torch.Tensor]:
    """
    Returns the EWC parameters of the model.
    """
    ewc_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            ewc_params[name] = param.detach().clone().to(device)
    return ewc_params
