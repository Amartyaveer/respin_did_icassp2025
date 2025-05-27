from tokenizers import Tokenizer
from transformers import RobertaModel, RobertaConfig
from torch import nn
import torch

class Embed(nn.Module):
    def __init__(self):
        super().__init__()
        config = RobertaConfig(num_hidden_layers=2, num_attention_heads=4, hidden_size=64, max_position_embeddings=4096)
        self.rberta = RobertaModel(config)
        
    def forward(self, x, attention_mask):
        x = self.rberta(input_ids=x, attention_mask=attention_mask).last_hidden_state 
        return x

class RobertaJointModel(nn.Module):
    def __init__(self, ctc, token_list, ln_out_dim, ignore_did_tokens, tokenizer_pth):
        super().__init__()
        self.ctc = ctc
        self.token_list = token_list
        self.embed = Embed()
        self.tokenizer = Tokenizer.from_file(tokenizer_pth)
        self.linear = nn.Linear(64, ln_out_dim)
        self.ignore_did_tokens = ignore_did_tokens

    def mean_pooling(self, hs_pad, hlens):
        """Apply mean pooling over the time dimension."""
        mask = (
            torch.arange(hs_pad.size(1), device=hs_pad.device)
            .expand(len(hlens), hs_pad.size(1))
            < hlens.unsqueeze(1)
        )
        hs_pad = hs_pad * mask.unsqueeze(-1)  # Zero out padded positions
        sum_hs_pad = hs_pad.sum(dim=1)  # Sum over time dimension
        length = mask.sum(dim=1, keepdim=True)  # Length of each sequence (non-padded)
        hs_pad_mean = sum_hs_pad / length  # Compute mean by dividing the sum by length
        return hs_pad_mean

    def cvt_alignment_to_text(self, ctc_alignment, token_list):
        ctc_alignment = [[token_list[token] for token in ctc_alignment[i].tolist()] for i in range(ctc_alignment.size(0))]
        ctc_text = []
        for alignment in ctc_alignment:
            words = []
            current_word = []
            for token in alignment:
                if token in ['<blank>', '<unk>', '<sos/eos>']:
                    continue
                if token == '<space>':
                    if current_word:
                        words.append("".join(current_word))
                        current_word = []  # reset for the next word
                else:
                    current_word.append(token)
            if current_word:
                words.append("".join(current_word))
            if not words:
                ctc_text.append('<blank>')
            else:
                ctc_text.append(" ".join(words))
        return ctc_text
    
    def forward(self, encoder_out, encoder_out_lens):
        device = encoder_out.device

        ctc_alignment = self.ctc.argmax(encoder_out, ignored_tokens=self.ignore_did_tokens)

        ctc_text = self.cvt_alignment_to_text(ctc_alignment, self.token_list)

        tokenized_out = [torch.tensor(self.tokenizer.encode(txt).ids).to(device) for txt in ctc_text]

        padded_out = torch.nn.utils.rnn.pad_sequence(tokenized_out, batch_first=True, padding_value=0).to(device)
        padded_out_lens = torch.tensor([len(txt) for txt in tokenized_out], device=device)
        padding_mask = torch.arange(padded_out.size(1), device=device).expand(len(padded_out_lens), padded_out.size(1)) < padded_out_lens.unsqueeze(1)
        
        ctc_embeddings = self.embed(padded_out, attention_mask=padding_mask).to(device)

        ctc_embeddings = self.mean_pooling(ctc_embeddings, padded_out_lens)

        ctc_rob_out = self.linear(ctc_embeddings).to(device)

        ctc_enc_out = self.ctc.ctc_lo(encoder_out).to(device)

        ctc_enc_out = ctc_enc_out[:, :, self.ignore_did_tokens]

        ctc_enc_out = self.mean_pooling(ctc_enc_out, encoder_out_lens)

        ctc_en_rob_out = torch.cat((ctc_rob_out, ctc_enc_out), dim=1).to(device)

        return ctc_en_rob_out
