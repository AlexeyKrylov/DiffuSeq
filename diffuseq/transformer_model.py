from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertModel, BertForMaskedLM
from transformers.models.deberta.modeling_deberta import DebertaEncoder, DebertaModel
import torch
from transformers import T5EncoderModel, RobertaForCausalLM
import torch as th
import torch.nn as nn

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
        args,
        input_dims,
        output_dims,
        hidden_t_dim,
        dropout=0,
        config=None,
        config_name='bert-base-cased',
        vocab_size=None,
        init_pretrained='no',
        logits_mode=1,
    ):
        super().__init__()

        self.args = args

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout

        self.config_name = config_name
        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)

        if args.token_type_embeddings:
            self.src_flag_emb = nn.Embedding(2, config.hidden_size)

        self.lm_head = nn.Linear(self.input_dims, vocab_size)

        if not self.args.trainable_lm_head:
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
        
        if init_pretrained == 't5_full':
            if args.debug_mode:
                print('initializing from pretrained t5...')
                print(config)

            embs = T5EncoderModel.from_pretrained(self.config_name)
            embs.set_input_embeddings(embs.shared)
            embs.resize_token_embeddings(self.args.vocab_size)
            self.word_embedding.from_pretrained(embs.shared.weight.clone(), freeze=False)

            del embs.encoder

            temp_t5 = T5EncoderModel(config)

            if not self.args.trainable_lm_head:
                with torch.no_grad():
                    self.lm_head.weight = self.word_embedding.weight

            self.input_transformers = temp_t5.encoder

            del temp_t5.shared
            del temp_t5.encoder.embed_tokens
            del embs.shared

        elif init_pretrained == 'bert_full':

            if args.debug_mode:
                print('initializing from pretrained bert...')
                print(config)

            temp_bert = BertForMaskedLM.from_pretrained(self.config_name)
            temp_bert.resize_token_embeddings(self.args.vocab_size)

            self.word_embedding = temp_bert.bert.embeddings.word_embeddings
            if not self.args.trainable_lm_head:
                with torch.no_grad():
                    self.lm_head.weight = temp_bert.bert.embeddings.word_embeddings.weight
            else:
                self.lm_head = temp_bert.cls

            self.input_transformers = temp_bert.bert.encoder

            self.register_buffer("position_ids", torch.arange(self.args.seq_len).expand((1, -1)))
            self.position_embeddings = temp_bert.bert.embeddings.position_embeddings

            if self.args.token_type_embeddings:
                self.src_flag_emb = temp_bert.bert.embeddings.token_type_embeddings

            self.LayerNorm = temp_bert.bert.embeddings.LayerNorm

            if not self.args.token_type_embeddings:
                del temp_bert.bert.embeddings.token_type_embeddings

        elif init_pretrained == 'roberta_full':

            if args.debug_mode:
                print('initializing from pretrained bert...')
                print(config)

            temp_bert = RobertaForCausalLM.from_pretrained(self.config_name, is_decoder=True)
            temp_bert.resize_token_embeddings(self.args.vocab_size)

            self.word_embedding = temp_bert.roberta.embeddings.word_embeddings
            if not self.args.trainable_lm_head:
                with torch.no_grad():
                    self.lm_head.weight = temp_bert.roberta.embeddings.word_embeddings.weight
            else:
                self.lm_head = temp_bert.lm_head

            self.input_transformers = temp_bert.roberta.encoder

            self.register_buffer("position_ids", torch.arange(514).expand((1, -1)))
            self.position_embeddings = temp_bert.roberta.embeddings.position_embeddings

            if self.args.token_type_embeddings:
                self.src_flag_emb = temp_bert.roberta.embeddings.token_type_embeddings

            self.LayerNorm = temp_bert.roberta.embeddings.LayerNorm

            if not self.args.token_type_embeddings:
                del temp_bert.roberta.embeddings.token_type_embeddings

        elif init_pretrained == 'no_bert+t5_emb':
            embs = T5EncoderModel.from_pretrained(self.config_name)
            embs.set_input_embeddings(embs.shared)
            embs.resize_token_embeddings(self.args.vocab_size)
            self.word_embedding.from_pretrained(embs.shared.weight.clone(), freeze=False)
            del embs.encoder
            del embs.shared

            if not self.args.trainable_lm_head:
                with torch.no_grad():
                    self.lm_head.weight = self.word_embedding.weight

            self.input_transformers = BertEncoder(config)

            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        elif init_pretrained == 'no_bert':
            self.input_transformers = BertEncoder(config)

            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        elif init_pretrained == 'no_t5':
            temp_t5 = T5EncoderModel(config)

            self.input_transformers = temp_t5.encoder

            del temp_t5.shared
            del temp_t5.encoder.embed_tokens
        else:
            assert False, "invalid type of init_pretrained"

        if init_pretrained != 't5_full':
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                  nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def forward(self, x, timesteps, attention_mask=None, input_ids_mask=None):
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

        if self.args.use_plm_init[:2] != "t5" and self.args.use_plm_init[:5] != "no_t5":
            position_ids = self.position_ids[:, : seq_length]
            emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        else:
            emb_inputs = emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)

        if self.args.token_type_embeddings:
            emb_src = self.src_flag_emb(input_ids_mask)
            emb_inputs += emb_src



        if self.args.use_plm_init[:2] != "t5" and self.args.use_plm_init[:5] != "no_t5":
            emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
            input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        else:
            input_trans_hidden_states = self.input_transformers(inputs_embeds=emb_inputs)[0]

        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        h = h.type(x.dtype)

        return h