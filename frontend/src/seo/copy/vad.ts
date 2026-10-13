import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Find where people are speaking in a recording, with start and end times for every speech segment.",
  about: [
    "Voice activity detection, or VAD, separates speech from silence and background noise. It is the step that decides which parts of a recording are worth sending to a transcriber, and the one that trims dead air from a voice note. Upload an audio file, record six seconds from your microphone or pick a sample, then press Detect speech.",
    "The result is a timeline rather than a label. Silero VAD v5 gives a speech probability for every 32 ms frame, drawn under the waveform on the same time axis, and a threshold slider turns those probabilities into speech segments listed with their start and end times. Dragging the threshold re-derives the segments from the scores already in hand, so the boundaries follow the pointer without the model running again. Gaps shorter than 100 ms are bridged and runs shorter than 250 ms are dropped, so a stop consonant does not split one phrase into several.",
    "Silero is tiny: 0.56 million parameters, a 2 MB download. It runs on your CPU through WebAssembly by design rather than as a fallback, at about 0.3 ms a frame, roughly a hundred times faster than real time, which leaves nothing for a GPU to win. Beside it is an energy detector that uses no model and downloads nothing, a baseline that measures loudness frame by frame, so the page is usable before any weights arrive. Audio stays in your browser.",
    "This is speech detection, not transcription and not speaker recognition: it says when someone is talking, not what they said or who they are. For the words, use the speech to text page.",
  ],
  aliases: [
    "voice activity detection",
    "Silero VAD",
    "speech detection",
    "speech segments",
    "VAD",
  ],
};
