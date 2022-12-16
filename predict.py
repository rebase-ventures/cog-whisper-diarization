"""
download the models to ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt  -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt  -P ./weights
"""

import io
import os
from typing import Optional, Any
import torch
import numpy as np
from cog import BasePredictor, Input, Path, BaseModel

import whisper
from whisper.model import Whisper, ModelDimensions
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisper.utils import format_timestamp

import ffmpeg
from pydub import AudioSegment

class ModelOutput(BaseModel):
    # Return an array of diarization segments
    diarization: Any
    #    start: float
    #    end: float
    #    speaker: str
    #    Plus regular whisper model output
    #    detected_language: str
    #    transcription: str
    #    segments: Any
    #    translation: Optional[str]
    #    txt_file: Optional[Path]
    #    srt_file: Optional[Path]

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.models = {}
        for model in ["tiny", "base", "small", "medium", "large-v1", "large-v2"]:
            with open(f"weights/{model}.pt", "rb") as fp:
                checkpoint = torch.load(fp, map_location="cpu")
                dims = ModelDimensions(**checkpoint["dims"])
                self.models[model] = Whisper(dims)
                self.models[model].load_state_dict(checkpoint["model_state_dict"])

    def predict(
        self,
        audio: Path = Input(description="Audio file"),
        model: str = Input(
            default="base",
            choices=["tiny", "base", "small", "medium", "large-v1", "large-v2"],
            description="Choose a Whisper model.",
        ),
        diarization: str = Input(
            default=None,
            description="JSON array of speaker diarization output from cog-pyannote model",
        ),
        transcription: str = Input(
            choices=["plain text", "srt", "vtt"],
            default="plain text",
            description="Choose the format for the transcription",
        ),
        translate: bool = Input(
            default=False,
            description="Translate the text to English when set to True",
        ),
        language: str = Input(
            choices=sorted(LANGUAGES.keys())
            + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
            default=None,
            description="language spoken in the audio, specify None to perform language detection",
        ),
        temperature: float = Input(
            default=0,
            description="temperature to use for sampling",
        ),
        patience: float = Input(
            default=None,
            description="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search",
        ),
        suppress_tokens: str = Input(
            default="-1",
            description="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations",
        ),
        initial_prompt: str = Input(
            default=None,
            description="optional text to provide as a prompt for the first window.",
        ),
        condition_on_previous_text: bool = Input(
            default=True,
            description="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop",
        ),
        temperature_increment_on_fallback: float = Input(
            default=0.2,
            description="temperature to increase when falling back when the decoding fails to meet either of the thresholds below",
        ),
        compression_ratio_threshold: float = Input(
            default=2.4,
            description="if the gzip compression ratio is higher than this value, treat the decoding as failed",
        ),
        logprob_threshold: float = Input(
            default=-1.0,
            description="if the average log probability is lower than this value, treat the decoding as failed",
        ),
        no_speech_threshold: float = Input(
            default=0.6,
            description="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence",
        ),
    ) -> ModelOutput:

        """Run a single prediction on the model"""
        print(f"Transcribe with {model} model")
        model = self.models[model].to("cuda")

        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        args = {
            "language": language,
            "patience": patience,
            "suppress_tokens": suppress_tokens,
            "initial_prompt": initial_prompt,
            "condition_on_previous_text": condition_on_previous_text,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
        }

        diarization_groups = group_diarization_segments_by_speaker(diarization)
        split_audio_file(str(audio), diarization_groups)

        for i, group in enumerate(diarization_groups):
            result = model.transcribe(str(i) + '.wav', temperature=temperature, **args)

            if transcription == "plain text":
                transcription = result["text"]
            elif transcription == "srt":
                transcription = write_srt(result["segments"])
            else:
                transcription = write_vtt(result["segments"])

            if translate:
                translation = model.transcribe(
                    str(i) + '.wav', task="translate", temperature=temperature, **args
                )
            # add results to diarization_groups
            diarization_groups[i]["segments"]=result["segments"],
            diarization_groups[i]["detected_language"]=LANGUAGES[result["language"]],
            diarization_groups[i]["transcription"]=transcription,
            diarization_groups[i]["translation"]=translation["text"] if translate else None,

        reutrn ModelOutput(diarization_groups)

def write_vtt(transcript):
    result = ""
    for segment in transcript:
        result += f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
        result += f"{segment['text'].strip().replace('-->', '->')}\n"
        result += "\n"
    return result


def write_srt(transcript):
    result = ""
    for i, segment in enumerate(transcript, start=1):
        result += f"{i}\n"
        result += f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
        result += f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
        result += f"{segment['text'].strip().replace('-->', '->')}\n"
        result += "\n"
    return result

def group_diarization_segments_by_speaker(diarization):
    new_diarization = []
    for i in range(len(diarization)):
        if i == 0:
            new_diarization.append({'start': diarization[i]['start'], 'end': diarization[i]['end'], 'speaker': diarization[i]['speaker']})
        elif diarization[i]['speaker'] == diarization[i-1]['speaker']:
            new_diarization[-1]['end'] = diarization[i]['end'] # update end time
        else:
            new_diarization.append({'start': diarization[i]['start'], 'end': diarization[i]['end'], 'speaker': diarization[i]['speaker']})
    return new_diarization

def extract_audio_as_wav(file, output_fn):
    # Convert input audio or video format to wav for pydub to split easily
    # Use same wav format as Wisper https://github.com/openai/whisper/blob/e90b8fa7e845ae184ed9aa0babcf3cde6f16719e/whisper/audio.py#L42
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output(output_fn, format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}")
    return True

def split_audio_file(wav_file, diarization_groups):
    audio = AudioSegment.from_wav(wav_file)
    # iterate through diarization groups with index
    for i, g in enumerate(diarization_groups):
        # convert start and end times to milliseconds
        start = int(g['start'] * 1000)
        end = int(g['end'] * 1000)
        audio[start:end].export(str(i) + '.wav', format='wav')
    return True