# Native English example

- Generated: 2026-10-12
- Source: [11-af-heart-english.txt](11-af-heart-english.txt)
- Output: [11-af-heart-english.wav](11-af-heart-english.wav)
- Voice: `af_heart`
- Model pack: local `models/kokoro-v1_0-af_heart.kpack`
- Frontend: eSpeak NG 1.52.0, invoked with `-q --ipa=3 -v en-us` and default rate and pitch
- Sample format: mono, PCM16, 24 kHz
- eSpeak output: 18 nonempty sentence lines, longest line 170 symbols before the 510-symbol model chunk limit
- Output duration: 57.525 seconds (1,380,600 samples)
- Generated with: `.build/kokoro say -f artifacts/11-af-heart-english.txt --output artifacts/11-af-heart-english.wav`

eSpeak NG's U+200D joiner markers are stripped because they are formatting characters and have no token in Kokoro's pinned vocabulary. Its U+0329 syllabic-consonant marker is rendered as a schwa before the consonant because the model vocabulary has no syllabic-consonant token. Other IPA symbols are mapped through the vocabulary used by the native reference path. This pronunciation can differ from Misaki. The WAV is saved for the listening acceptance in ticket 14; no human listening judgment is recorded here.
