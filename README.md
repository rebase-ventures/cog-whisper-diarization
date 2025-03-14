# Cog Whisper model which includes speaker diarization

Use cog-pyannote model on file first to get speaker diarization JSON array. This model then takes this input of timings, groups by speaker and splits input file into segments, before running Whisper like normal on each speaker segment separately.

`cog build -t whisper-diarization`
We need to use `--add-host host.docker.internal:host-gateway` so that docker can access localhost where mini_httpd serves up a local data directory. We also serve this model on 5001 as cog-pyannote (speaker-diarization) is running on 5000.
`docker run -d -p 5001:5000 --add-host host.docker.internal:host-gateway --gpus all whisper-diarization`

Example running with cli:
`cog predict -i "model='large-v2' audio='@jre-kevin-hart-youtube-short-vzx6h2sAGTU.wav' diarization='[{\"start\": 0.4978125, \"end\": 5.442187499999999, \"speaker\": \"SPEAKER_01\"}, {\"start\": 5.442187499999999, \"end\": 8.513437500000002, \"speaker\": \"SPEAKER_00\"}, {\"start\": 9.424687500000001, \"end\": 17.5078125, \"speaker\": \"SPEAKER_00\"}, {\"start\": 9.6440625, \"end\": 10.572187500000002, \"speaker\": \"SPEAKER_01\"}, {\"start\": 11.4328125, \"end\": 12.630937500000002, \"speaker\": \"SPEAKER_01\"}, {\"start\": 17.5078125, \"end\": 47.3090625, \"speaker\": \"SPEAKER_01\"}, {\"start\": 44.8959375, \"end\": 56.8096875, \"speaker\": \"SPEAKER_00\"}, {\"start\": 58.10906250000001, \"end\": 63.1378125, \"speaker\": \"SPEAKER_00\"}, {\"start\": 63.81281250000001, \"end\": 102.3384375, \"speaker\": \"SPEAKER_00\"}, {\"start\": 102.3384375, \"end\": 105.4603125, \"speaker\": \"SPEAKER_01\"}]'"`

# Whisper

[[Blog]](https://openai.com/blog/whisper)
[[Paper]](https://cdn.openai.com/papers/whisper.pdf)
[[Model card]](model-card.md)
[[Colab example]](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)
[![Replicate](https://replicate.com/openai/whisper/badge)](https://replicate.com/openai/whisper)

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.


## Approach

![Approach](approach.png)

A Transformer sequence-to-sequence model is trained on various speech processing tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. All of these tasks are jointly represented as a sequence of tokens to be predicted by the decoder, allowing for a single model to replace many different stages of a traditional speech processing pipeline. The multitask training format uses a set of special tokens that serve as task specifiers or classification targets.


## Setup

We used Python 3.9.9 and [PyTorch](https://pytorch.org/) 1.10.1 to train and test our models, but the codebase is expected to be compatible with Python 3.7 or later and recent PyTorch versions. The codebase also depends on a few Python packages, most notably [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) for their fast tokenizer implementation and [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) for reading audio files. The following command will pull and install the latest commit from this repository, along with its Python dependencies 

    pip install git+https://github.com/openai/whisper.git 

To update the package to the latest version of this repository, please run:

    pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git

It also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

You may need [`rust`](http://rust-lang.org) installed as well, in case [tokenizers](https://pypi.org/project/tokenizers/) does not provide a pre-built wheel for your platform. If you see installation errors during the `pip install` command above, please follow the [Getting started page](https://www.rust-lang.org/learn/get-started) to install Rust development environment. Additionally, you may need to configure the `PATH` environment variable, e.g. `export PATH="$HOME/.cargo/bin:$PATH"`. If the installation fails with `No module named 'setuptools_rust'`, you need to install `setuptools_rust`, e.g. by running:

```bash
pip install setuptools-rust
```


## Available models and languages

There are five model sizes, four with English-only versions, offering speed and accuracy tradeoffs. Below are the names of the available models and their approximate memory requirements and relative speed. 


|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~32x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~16x      |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~6x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |

For English-only applications, the `.en` models tend to perform better, especially for the `tiny.en` and `base.en` models. We observed that the difference becomes less significant for the `small.en` and `medium.en` models.

Whisper's performance varies widely depending on the language. The figure below shows a WER breakdown by languages of Fleurs dataset, using the `large` model. More WER and BLEU scores corresponding to the other models and datasets can be found in Appendix D in [the paper](https://cdn.openai.com/papers/whisper.pdf).

![WER breakdown by language](language-breakdown.svg)


## More examples

Please use the [🙌 Show and tell](https://github.com/openai/whisper/discussions/categories/show-and-tell) category in Discussions for sharing more example usages of Whisper and third-party extensions such as web demos, integrations with other tools, ports for different platforms, etc.


## License

The code and the model weights of Whisper are released under the MIT License. See [LICENSE](LICENSE) for further details.
