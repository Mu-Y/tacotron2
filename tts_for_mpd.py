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


def load_tts_vocoder_models(tacotron_checkpoint_path, waveglow_checkpoint_path):
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


# def parse_input(input_text_file_path):
#     with open(input_text_file_path, 'r') as f:
#         lines = f.readlines()
#     phoneme_pairs = []
#     texts = []
#     for line in lines:
#         items = line.strip().split('|')
#         phoneme_pairs.append(items[0].split())
#         texts.append(items[1])
#     return phoneme_pairs, texts

def parse_input(input_text_file_path):
    with open(input_text_file_path, 'r') as f:
        lines = f.readlines()
    texts = []
    for line in lines:
        texts.append(line.strip())
    return texts


def text_to_arpabet(cmu_dict, phoneme_pairs, text, swap_phoneme=False):
    text = re.sub(r'[^\w ]', '', text)
    text_arpabet_list = []
    for word in text.split():
        arpabet = cmu_dict.lookup(word)[0]
        phoneme_list = []
        has_interested_phoneme = False
        for phoneme in arpabet.split():
            if phoneme == phoneme_pairs[0]:
                has_interested_phoneme = True
                if swap_phoneme:
                    phoneme_list.append(phoneme_pairs[1])
                else:
                    phoneme_list.append(phoneme)
            elif phoneme == phoneme_pairs[1]:
                has_interested_phoneme = True
                if swap_phoneme:
                    phoneme_list.append(phoneme_pairs[0])
                else:
                    phoneme_list.append(phoneme)
            else:
                phoneme_list.append(phoneme)
        if not has_interested_phoneme:
            text_arpabet_list.append(word)
        else:
            text_arpabet_list.append('{%s}' % ' '.join(phoneme_list))
    return ' '.join(text_arpabet_list)


def main(args):
    model, waveglow, denoiser, hparams = load_tts_vocoder_models(args.tacotron_checkpoint_path, args.waveglow_checkpoint_path)
    cmu_dict = load_cmudict(args.cmudict_path)
    # phoneme_pairs, texts = parse_input(args.input_text_file)
    texts = parse_input(args.input_text_file)

    phoneme_pairs = ['W V', 'IH0 IY0', 'IH1 IY1', 'IH2 IY2', 'EH0 AE0', 'EH1 AE1', 'EH2 AE2', 'NG N', 'S TH', 'Z S',
                     'AA0 AH0', 'AA1 AH1', 'AA2 AH2', 'UW0 UH0', 'UW1 UH1', 'UW2 UH2', 'DH D']

    # assert len(phoneme_pairs) == len(texts), "Lines of phoneme pairs and texts must be the same, please check" \
    #                                          "the input text file."

    wav_dir = args.output_dir.joinpath('wav')
    wav_dir.mkdir(exist_ok=True, parents=True)
    trans_dir = args.output_dir.joinpath('transcript')
    trans_dir.mkdir(exist_ok=True, parents=True)
    tg_dir = args.output_dir.joinpath('annotation')
    tg_dir.mkdir(exist_ok=True, parents=True)

    utt_i = 0
    for text in texts:
        for phoneme_pair in phoneme_pairs:
            # Synthesize speech
            phoneme_pair = phoneme_pair.split()
            try:
                text_arpabet = text_to_arpabet(cmu_dict, phoneme_pair, text, swap_phoneme=args.mispronunciation)
            except:
                continue
            if text_arpabet == re.sub(r'[^\w ]', '', text).strip():
                continue

            sequence = np.array(text_to_sequence(text_arpabet, ['english_cleaners']))[None, :]
            sequence = torch.autograd.Variable(
                torch.from_numpy(sequence)).cuda().long()
            _, mel_outputs_postnet, _, alignments, is_max_steps = model.inference(sequence)
            if is_max_steps:
                continue
            with torch.no_grad():
                wav = waveglow.infer(mel_outputs_postnet, sigma=0.666)
            wav_denoised = denoiser(wav, strength=0.01)[:, 0].cpu().numpy().T
            output_wav_file = wav_dir.joinpath('{:s}_{:04d}.wav'.format(args.prefix, utt_i + 1))
            wavfile.write(output_wav_file, hparams.sampling_rate, wav_denoised)

            # Save transcript
            output_trans_file = trans_dir.joinpath('{:s}_{:04d}.txt'.format(args.prefix, utt_i + 1))
            with open(output_trans_file, 'w') as f:
                f.write(text)

            # Generate textgrid
            tg = textgrid.TextGrid()
            word_tier = textgrid.IntervalTier(name='words')
            text = re.sub(r'[^\w]', ' ', text)
            idx = 0
            for word in text.split():
                word_tier.add(float(idx), float(idx + 1), word)
                idx += 1

            idx = 0
            phone_tier = textgrid.IntervalTier(name='phones')
            phone_tier.add(float(idx), float(idx + 1), 'sil')
            idx += 1
            for word in text.split():
                arpabet = cmu_dict.lookup(word)[0]
                for phoneme in arpabet.split():
                    if args.mispronunciation:
                        if phoneme == phoneme_pair[0]:
                            phone_tier.add(float(idx), float(idx + 1), phoneme + ',' + phoneme_pair[1] + ',s')
                        elif phoneme == phoneme_pair[1]:
                            phone_tier.add(float(idx), float(idx + 1), phoneme + ',' + phoneme_pair[0] + ',s')
                        else:
                            phone_tier.add(float(idx), float(idx + 1), phoneme)
                    else:
                        phone_tier.add(float(idx), float(idx + 1), phoneme)
                    idx += 1    
            
            phone_tier.add(float(idx), float(idx + 1), 'sil')
            tg.append(word_tier)
            tg.append(phone_tier)

            tg_file = tg_dir.joinpath('{:s}_{:04d}.TextGrid'.format(args.prefix, utt_i + 1))
            tg.write(tg_file)

            print('{:d}: {:s} | {:s} | {:s}'.format(utt_i + 1, text, ' '.join(phoneme_pair), text_arpabet))
            utt_i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_text_file", type=Path, help= "Path to the input to TTS system."
                                                           "Each column of it should be:"
                                                           "Phoneme pair|Text"
                                                           "For example:"
                                                           "V W|Very Well"
                                                           "If --mispronunciation is set, the synthesis will have"
                                                           "mispronunciations by swapping the two phonemes.")
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
    parser.add_argument("--mispronunciation", action="store_true", help= \
        "Set to True if you would like to create intentional mispronunciations.")
    args = parser.parse_args()

    main(args)
