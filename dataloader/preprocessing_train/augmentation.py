from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift


# Audio augmentation object to map over the dataset:
AUGMENT_WAVEFORM = Compose([
    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=0.3),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3, leave_length_unchanged=False),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.3)
])


def augment_audio_fct(batch, sample_rate: int):
    """
    [NOT TESTED YET] Perform data augmentation for audio.
    
    Notes:
        - `extract_audio` must be called before this function
        - should only be applied to the train set
    """
    audio = batch["audio"]["array"]
    augmented_audio = AUGMENT_WAVEFORM(samples=audio, sample_rate=sample_rate)
    batch["audio"]["array"] = augmented_audio
    return batch
