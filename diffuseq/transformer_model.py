from transformers import AutoConfig
# from transformers import BertEncoder
from transformers.models.bert.modeling_bert import BertEncoder, BertModel
import torch
from transformers import T5ForConditionalGeneration

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)


class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
            self,
            input_dims,
            output_dims,
            hidden_t_dim,
            dropout=0,
            config=None,
            config_name='bert-base-uncased',
            vocab_size=None,
            init_pretrained='no',
            logits_mode=1,
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)
        self.lm_head = nn.Linear(self.input_dims, vocab_size)

        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
                                               nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))

        if init_pretrained == 'bert':
            print('initializing from pretrained bert...')
            print(config)
            temp_bert = BertModel.from_pretrained("prajjwal1/bert-tiny")

            embedding_layer = temp_bert.embeddings.word_embeddings
            old_num_tokens, old_embedding_dim = embedding_layer.weight.shape
            new_embeddings = nn.Embedding(33913, old_embedding_dim)  # TODO

            new_embeddings.to(embedding_layer.weight.device, dtype=embedding_layer.weight.dtype)
            new_embeddings.weight.data[:old_num_tokens, :] = embedding_layer.weight.data[:old_num_tokens, :]

            self.word_embedding = new_embeddings

            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight

            # self.lm_head.weight.requires_grad = False
            # self.word_embedding.weight.requires_grad = False

            self.input_transformers = BertEncoder(config)
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            # self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            # self.LayerNorm = temp_bert.embeddings.LayerNorm
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

            del temp_bert.embeddings
            del temp_bert.pooler

        elif init_pretrained == 'no':

            self.input_transformers = BertEncoder(config)

            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        elif init_pretrained == 'T5':
            input_transformers = T5ForConditionalGeneration.from_pretrained(config_name)

            self.encoder = input_transformers.encoder
            self.decoder = input_transformers.decoder
            # self.input_transformers.resize_token_embeddings(35584)

            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
            del self.encoder.embed_tokens
            del self.decoder.embed_tokens
        else:
            assert False, "invalid type of init_pretrained"

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                  nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:  # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1))  # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        kind_of_type = x.dtype
        emb_t_0 = timestep_embedding(timesteps, self.hidden_t_dim)
        emb_t_0 = emb_t_0.to(kind_of_type)
        emb_t = self.time_embed(emb_t_0)

        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        seq_length = x.size(1)

        emb_inputs = emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)

        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))


        emb_inputs_1, emb_inputs_2 = emb_inputs.chunk(2, dim=1)

        hidden_states = self.encoder(
            inputs_embeds=emb_inputs_1
        )[0]


        decoder_outputs = self.decoder(
            inputs_embeds=emb_inputs_2,
            encoder_hidden_states=hidden_states
        )[0]

        # input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        input_trans_hidden_states = torch.concat([hidden_states, decoder_outputs], axis=1)
        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        h = h.type(x.dtype)
        return h