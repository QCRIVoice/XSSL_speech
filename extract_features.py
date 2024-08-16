import argparse
import json
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, WavLMForCTC, HubertForCTC
from tqdm import tqdm

class Embedder:
    def __init__(self, model_name, dataset, model_type='wav2vec'):
        self.model_name = model_name
        self.dataset = dataset
        self.model_type = model_type

        if self.model_type == 'wav2vec':
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name, output_hidden_states=True, output_attentions=True)
        elif self.model_type == 'hubert':
            self.model = HubertForCTC.from_pretrained(self.model_name, output_hidden_states=True, output_attentions=True)
        elif self.model_type == 'wavlm':
            self.model = WavLMForCTC.from_pretrained(self.model_name, output_hidden_states=True, output_attentions=True)

        self.model.to('cuda')
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained('facebook/wav2vec2-base-960h')

    def get_hidden_states(self, audio):
        inputs = self.tokenizer(audio, return_tensors="pt", padding="longest")
        with torch.no_grad():
            hidden_states = self.model(inputs.input_values[0].to('cuda'))
            hidden_states_logits = hidden_states['hidden_states']
            hidden_states_attention = hidden_states['attentions']
        
        return hidden_states_logits, hidden_states_attention

    def get_all(self, audio_list, name):
        hidden_states_list = [{} for _ in range(25)]
        hidden_states_list_attention = [{} for _ in range(24)]

        for audio_path in tqdm(audio_list, desc="Processing audio files"):
            print(f"Processing: {audio_path}")
            waveform, sample_rate = torchaudio.load(audio_path)
            hidden_states, attention_states = self.get_hidden_states(waveform)
            
            hidden_states_list[24][audio_path] = hidden_states[24][0].cpu().numpy()
            hidden_states_list_attention[23][audio_path] = attention_states[23].cpu().numpy()

        np.savez(f'XSpeech_SSL/outputs/embds/{name}_24_lp.npz', **hidden_states_list[24])
        np.savez(f'XSpeech_SSL/outputs/embds/{name}_23_att.npz', **hidden_states_list_attention[23])

        return hidden_states_list, hidden_states_list_attention

def main(args):
    with open(args.alignment_file, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        audio_keys = list(data.keys())

    embedder = Embedder(args.model_name, args.dataset, args.model_type)
    hidden_states_emb = embedder.get_all(audio_keys, args.output_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hidden states from audio files using a pretrained model.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the pretrained model.")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument('--alignment_file', type=str, required=True, help="Path to the JSON alignment fil, or just path files.")
    parser.add_argument('--output_name', type=str, required=True, help="Name for the output NPZ files.")
    parser.add_argument('--model_type', type=str, default='wav2vec', choices=['wav2vec', 'hubert', 'wavlm'], help="Type of model to use.")
    
    args = parser.parse_args()
    main(args)
