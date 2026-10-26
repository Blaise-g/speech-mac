# Speech generation

Terms for local English speech generation in speech-mac.

## Language

**Frontend**:
Normalization and tokenization of ordinary English text for speech generation.

**Inference**:
Generation of audio from prepared text, a compatible voice and generation controls.

**Voice**:
A compatible local artifact that conditions the speaker identity of generated speech.
_Avoid_: Style

**Resident engine**:
A speech engine that keeps model weights loaded across synthesis requests.

**Progressive playback**:
Playback of generated audio frames while later frames of the same request are still being generated.

**Duration delta**:
The model's native duration conditioning control.
_Avoid_: Speed
