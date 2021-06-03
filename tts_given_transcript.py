import re
import argparse
import numpy as np
import torch
from pathlib import Path

import sys
sys.path.append('waveglow/')


from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence, cmudict
from waveglow.denoiser import Denoiser
from scipy.io import wavfile
import textgrid
from collections import defaultdict
import pdb
import os


def load_tts_vocoder_models(tacotron_checkpoint_path, waveglow_checkpoint_path):
    '''
    Tacotron model was trained with mixed grapheme and phonemes.
    '''
    hparams = create_hparams()
    hparams.sampling_rate = 22050

    model = load_model(hparams)
    model.load_state_dict(torch.load(tacotron_checkpoint_path)['state_dict'])
    _ = model.cuda().eval()


    waveglow = torch.load(waveglow_checkpoint_path)['model']
    waveglow.cuda().eval()
    #for k in waveglow.convinv:
    #    k.float()
    denoiser = Denoiser(waveglow)
    return model, waveglow, denoiser, hparams


def load_cmudict(cmudict_path):
    if not cmudict_path.is_file():
        raise Exception('If use_cmudict=True, you must download ' +
                        'http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b to %s' % cmudict_path)
    d = cmudict.CMUDict(str(cmudict_path), keep_ambiguous=True)
    print('Loaded CMUDict with %d unambiguous entries' % len(d))
    return d

def get_phones_from_tier(phone_tier, selection='canonical'):
    '''
    selection: canonical or perceived
    '''
    phones = []
    for phone_interval in phone_tier.intervals:
        items = phone_interval.mark.split(',')
        if len(items) > 1:
            if selection=='canonical':
                # phones.append(re.sub(r' ', '', items[0]))
                phones.append(items[0])
            elif selection=='perceived':
                # phones.append(re.sub(r' ', '', items[1]))
                phones.append(items[1])
        else:
            # phones.append(re.sub(r'[^\w ]', '', phone_interval.mark))
            phones.append(phone_interval.mark)
    return ' '.join(phones)


def parse_input(input_dataset_dir: Path, speakers=['LXC']):
    '''
    Inputs:
        input_dataset_dir: L2ARCTIC directory by default. /mnt/data1/shaojin/L2ARCTIC
        speakers: the speaker which provides transcripts.
    Outputs:
        text: othographical transcripts. E.g., Today is sunny.
        text_arpabet: words for correctly pronounced words, phonemes for mispronounced words. E.g. Today is {S AA N IY}.
        metadata: dict of {tg_path: {tg:.TextGrid file
                                     text:transcription string,
                                     arpabet_canonical:arpabet string (canonical),
                                     arpabet_perceived:arpabet string (perceived),
                                     text_arpabet:mixed grapheme-arpabet string (input to shaojin TTS)}}
    '''
    metadata = defaultdict(dict)
    for speaker in speakers:
        speaker_dir = input_dataset_dir.joinpath(speaker)
        tg_dir = speaker_dir.joinpath('annotation')
        for tg_path in tg_dir.glob('*.TextGrid'):
            tg = textgrid.TextGrid()
            try:
                tg.read(tg_path)
            except ValueError:
                continue

            word_tier = tg.getFirst('words')
            phone_tier = tg.getFirst('phones')
            text, text_arpabet = [], []
            for word_interval in word_tier.intervals:
                if len(word_interval.mark) == 0:
                    continue
                is_mispronuounced = False
                for phone_interval in phone_tier.intervals:
                    if not (phone_interval.minTime >= word_interval.minTime and
                            phone_interval.maxTime <= word_interval.maxTime):
                        continue
                    if len(phone_interval.mark.split(',')) > 1 and re.sub(r' ', '',
                                                                          phone_interval.mark.split(',')[-1]) == 's':
                        is_mispronuounced = True
                # is_mispronuounced = True
                if not is_mispronuounced:
                    text_arpabet.append(word_interval.mark)
                else:
                    phones = []
                    for phone_interval in phone_tier.intervals:
                        if not (phone_interval.minTime >= word_interval.minTime and
                                phone_interval.maxTime <= word_interval.maxTime):
                            continue
                        items = phone_interval.mark.split(',')
                        if len(items) > 1:
                            if re.sub(r' ', '', items[-1]) == 's':
                                if re.sub(r' ', '', items[1]) == 'err':
                                    phones.append(re.sub(r'[\W ]', '', items[0]))
                                elif items[0][-1].isdigit() and not items[1][-1].isdigit():
                                    phones.append(re.sub(r'[\W ]', '', items[1] + items[0][-1]))
                                else:
                                    phones.append(re.sub(r'[\W ]', '', items[1]))
                            elif re.sub(r' ', '', items[-1]) == 'd':
                                phones.append(re.sub(r'[\W ]', '', items[0]))
                            else:
                                continue
                        else:
                            phones.append(re.sub(r'[\W ]', '', phone_interval.mark))
                    text_arpabet.append('{%s}' % ' '.join(phones))
                text.append(word_interval.mark)

            metadata[str(tg_path)]['tg'] = tg
            metadata[str(tg_path)]['text'] = ' '.join(text)
            metadata[str(tg_path)]['text_arpabet'] = ' '.join(text_arpabet)
            metadata[str(tg_path)]['arpabet_canonical'] = get_phones_from_tier(phone_tier, 'canonical')
            metadata[str(tg_path)]['arpabet_perceived'] = get_phones_from_tier(phone_tier, 'perceived')
    return metadata

def write_metadata_txt(metadata, output_dir):
    with output_dir.joinpath('metadata.txt').open('w') as f:
        for x in sorted(metadata):
            tg_path = x
            text = metadata[x]['text']
            text_arpabet = metadata[x]['text_arpabet']
            arpabet_canonical = metadata[x]['arpabet_canonical']
            arpabet_perceived = metadata[x]['arpabet_perceived']
            line = '{}|{}|{}|{}|{}\n'.format(tg_path, text, text_arpabet, arpabet_canonical, arpabet_perceived)
            f.write(line)
    print('metadata written to {}'.format(str(output_dir.joinpath('metadata.txt'))))


def main(args):
    model, waveglow, denoiser, hparams = load_tts_vocoder_models(args.tacotron_checkpoint_path, args.waveglow_checkpoint_path)

    # phoneme_pairs, texts = parse_input(args.input_text_file)

    # texts, texts_arpabet, tgs = parse_input(args.input_dataset_dir)
    metadata = parse_input(args.input_dataset_dir)
    texts = [metadata[x]['text'] for x in sorted(metadata)]
    texts_arpabet = [metadata[x]['text_arpabet'] for x in sorted(metadata)]
    tgs = [metadata[x]['tg'] for x in sorted(metadata)]

    wav_dir = args.output_dir.joinpath('wav')
    wav_dir.mkdir(exist_ok=True, parents=True)
    trans_dir = args.output_dir.joinpath('transcript')
    trans_dir.mkdir(exist_ok=True, parents=True)
    tg_dir = args.output_dir.joinpath('annotation')
    tg_dir.mkdir(exist_ok=True, parents=True)

    # store the inputs to TTS, to know what's going on/being synthesized
    write_metadata_txt(metadata, args.output_dir)

    for tg_path in sorted(metadata):
        text_arpabet = metadata[tg_path]['text_arpabet']
        text = metadata[tg_path]['text']
        sequence = np.array(text_to_sequence(text_arpabet, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()
        _, mel_outputs_postnet, _, alignments, is_max_steps = model.inference(sequence)
        if is_max_steps:
            continue
        with torch.no_grad():
            wav = waveglow.infer(mel_outputs_postnet, sigma=0.666)
        # wav_denoised = denoiser(wav, strength=0.01)[:, 0].cpu().numpy().T

        # get the base filename, without the .TextGrid extension
        base_name = os.path.basename(tg_path).split(".")[0]
        output_wav_file = wav_dir.joinpath('{:s}_{}.wav'.format(args.prefix, base_name))
        wavfile.write(output_wav_file, hparams.sampling_rate, wav.cpu().numpy().T)

        # Save transcript
        output_trans_file = trans_dir.joinpath('{:s}_{}.txt'.format(args.prefix, base_name))
        with open(output_trans_file, 'w') as f:
            f.write(text)

        # Generate textgrid
        tg = metadata[tg_path]['tg']

        tg_file = tg_dir.joinpath('{:s}_{}.TextGrid'.format(args.prefix, base_name))
        tg.write(tg_file)

        print('{}: {:s} | {:s}'.format(base_name, text, text_arpabet))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dataset_dir", type=Path, help= "L2ARCTIC directory.")
    parser.add_argument("output_dir", type=Path, help="Directory of synthesis")
    parser.add_argument("prefix", type=str, help="Prefix of the files")
    parser.add_argument("--tacotron_checkpoint_path", type=Path,
                        default="pretrained_models/checkpoint_42000", help= \
                            "Checkpoint path of tacotron model.")
    parser.add_argument("--waveglow_checkpoint_path", type=Path,
                        default="pretrained_models/waveglow_256channels_universal_v5.pt", help= \
                            "Checkpoint path of waveglow model.")
    parser.add_argument("--cmudict_path", type=Path,
                        default="pretrained_models/cmudict-0.7b", help= \
                            "Path to cmu dictionary.")
    args = parser.parse_args()

    main(args)
