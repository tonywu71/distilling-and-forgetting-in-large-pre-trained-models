from transformers import TrainerCallback, TrainerState

class EvalFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state: TrainerState, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True
