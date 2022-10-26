import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
    BertLMPredictionHead,
)
from transformers.activations import gelu
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel,
    RobertaLMHead,
)


def align_loss_fct(x, y, alpha=2):  # from Wang and Isola, 2018
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config, linear=False):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        if not linear:  # add non-linear activation
            self.activation = nn.Tanh()
        else:
            self.activation = None

    def forward(self, features, **kwargs):
        x = self.dense1(features)
        if self.activation is not None:
            x = self.activation(x)
            x = self.dense2(x)
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls",
            "cls_before_pooler",
            "avg",
            "avg_top2",
            "avg_first_last",
        ], (
            "unrecognized pooling type %s" % self.pooler_type
        )

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ["cls_before_pooler", "cls"]:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.c_sim = Similarity(temp=cls.model_args.temp)
    cls.a_sim = Similarity(temp=0.05)
    cls.init_weights()


def normalize(z0, z1):
    return (
        torch.matmul(z0, z0.T) + torch.matmul(z1, z1.T) - 2 * torch.matmul(z0, z1.T)
    ).mean()


def cl_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    linear=None,
    bin_mask=None,
):
    mlm_outputs = None
    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1)
    max_len = input_ids.size(-1)
    device = encoder.device

    if token_type_ids is not None:  # e.g. for roberta
        orig_z0_input_ids, orig_z0_mask, orig_z0_tok_type = (
            input_ids[:, 0, :],
            attention_mask[:, 0, :],
            token_type_ids[:, 0, :],
        )
        orig_z1_input_ids, orig_z1_mask, orig_z1_tok_type = (
            input_ids[:, 1, :],
            attention_mask[:, 1, :],
            token_type_ids[:, 1, :],
        )
        aug_z0_input_ids, aug_z0_mask, aug_z0_tok_type = (
            input_ids[:, 2, :],
            attention_mask[:, 2, :],
            token_type_ids[:, 2, :],
        )
        aug_z1_input_ids, aug_z1_mask, aug_z1_tok_type = (
            input_ids[:, 3, :],
            attention_mask[:, 3, :],
            token_type_ids[:, 3, :],
        )

        orig_z0_outputs = encoder(
            input_ids=orig_z0_input_ids,
            attention_mask=orig_z0_mask,
            token_type_ids=orig_z0_tok_type,
            return_dict=True,
        )

        orig_z1_outputs = encoder(
            input_ids=orig_z1_input_ids,
            attention_mask=orig_z1_mask,
            token_type_ids=orig_z1_tok_type,
            return_dict=True,
        )

        aug_z0_outputs = encoder(
            input_ids=aug_z0_input_ids,
            attention_mask=aug_z0_mask,
            token_type_ids=aug_z0_tok_type,
            return_dict=True,
        )

        aug_z1_outputs = encoder(
            input_ids=aug_z1_input_ids,
            attention_mask=aug_z1_mask,
            token_type_ids=aug_z1_tok_type,
            return_dict=True,
        )
    else:  # e.g. for bert
        orig_z0_input_ids, orig_z0_mask = input_ids[:, 0, :], attention_mask[:, 0, :]
        orig_z1_input_ids, orig_z1_mask = input_ids[:, 1, :], attention_mask[:, 1, :]
        aug_z0_input_ids, aug_z0_mask = input_ids[:, 2, :], attention_mask[:, 2, :]
        aug_z1_input_ids, aug_z1_mask = input_ids[:, 3, :], attention_mask[:, 3, :]
        orig_z0_outputs = encoder(
            input_ids=orig_z0_input_ids,
            attention_mask=orig_z0_mask,
            return_dict=True,
        )
        orig_z1_outputs = encoder(
            input_ids=orig_z1_input_ids,
            attention_mask=orig_z1_mask,
            return_dict=True,
        )
        aug_z0_outputs = encoder(
            input_ids=aug_z0_input_ids,
            attention_mask=aug_z0_mask,
            return_dict=True,
        )
        aug_z1_outputs = encoder(
            input_ids=aug_z1_input_ids,
            attention_mask=aug_z1_mask,
            return_dict=True,
        )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        attention_mask_flat = attention_mask.view((-1, attention_mask.size(-1)))
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask_flat,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True
            if cls.model_args.pooler_type in ["avg_top2", "avg_first_last"]
            else False,
            return_dict=True,
        )

    orig_z0 = cls.pooler(orig_z0_mask, orig_z0_outputs)
    orig_z1 = cls.pooler(orig_z1_mask, orig_z1_outputs)
    aug_z0 = cls.pooler(aug_z0_mask, aug_z0_outputs)
    aug_z1 = cls.pooler(aug_z1_mask, aug_z1_outputs)

    if cls.pooler_type == "cls":
        orig_z0 = cls.mlp(orig_z0)
        orig_z1 = cls.mlp(orig_z1)
        aug_z0 = cls.mlp(aug_z0)
        aug_z1 = cls.mlp(aug_z1)

    if num_sent == 6:
        hard_orig_input_ids, hard_orig_mask = (
            input_ids[:, 4, :],
            attention_mask[:, 4, :],
        )
        hard_aug_input_ids, hard_aug_mask = input_ids[:, 5, :], attention_mask[:, 5, :]
        hard_orig_outputs = encoder(
            input_ids=hard_orig_input_ids,
            attention_mask=hard_orig_mask,
            return_dict=True,
        )
        hard_aug_outputs = encoder(
            input_ids=hard_aug_input_ids,
            attention_mask=hard_aug_mask,
            return_dict=True,
        )
        hard_orig = cls.pooler(hard_orig_mask, hard_orig_outputs)
        hard_aug = cls.pooler(hard_aug_mask, hard_aug_outputs)

        if cls.pooler_type == "cls":
            hard_orig = cls.mlp(hard_orig)
            hard_aug = cls.mlp(hard_aug)

    if bin_mask is not None:
        aug_z0_mask = torch.gt(bin_mask[:, 2], 0)

    if dist.is_initialized() and cls.training:
        # Gather hard negative - not used in paper
        if num_sent == 6:
            hard_orig_list = [
                torch.zeros_like(hard_orig) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(tensor_list=hard_orig_list, tensor=hard_orig.contiguous())
            hard_orig_list[dist.get_rank()] = hard_orig
            hard_orig = torch.cat(hard_orig_list, 0)

            hard_aug_list = [
                torch.zeros_like(hard_aug) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(tensor_list=hard_aug_list, tensor=hard_aug.contiguous())
            hard_aug_list[dist.get_rank()] = hard_aug
            hard_aug = torch.cat(hard_aug_list, 0)

        # Dummy vectors for allgather
        orig_z0_list = [torch.zeros_like(orig_z0) for _ in range(dist.get_world_size())]
        orig_z1_list = [torch.zeros_like(orig_z1) for _ in range(dist.get_world_size())]
        aug_z0_list = [torch.zeros_like(aug_z0) for _ in range(dist.get_world_size())]
        aug_z1_list = [torch.zeros_like(aug_z1) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=orig_z0_list, tensor=orig_z0.contiguous())
        dist.all_gather(tensor_list=orig_z1_list, tensor=orig_z1.contiguous())
        dist.all_gather(tensor_list=aug_z0_list, tensor=aug_z0.contiguous())
        dist.all_gather(tensor_list=aug_z1_list, tensor=aug_z1.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        orig_z0_list[dist.get_rank()] = orig_z0
        orig_z1_list[dist.get_rank()] = orig_z1
        aug_z0_list[dist.get_rank()] = aug_z0
        aug_z1_list[dist.get_rank()] = aug_z1
        # Get full batch embeddings: (bs x N, hidden)
        orig_z0 = torch.cat(orig_z0_list, 0)
        orig_z1 = torch.cat(orig_z1_list, 0)
        aug_z0 = torch.cat(aug_z0_list, 0)
        aug_z1 = torch.cat(aug_z1_list, 0)

    gend_z0 = torch.cat((orig_z0, aug_z0))
    gend_z1 = torch.cat((orig_z1, aug_z1))

    c_loss = None
    a_loss = None

    if cls.model_args.cl_loss:
        z0 = torch.cat((orig_z0, aug_z0))
        z1 = torch.cat((orig_z1, aug_z1))
        cos_sim = cls.c_sim(z0.unsqueeze(1), z1.unsqueeze(0))
        if num_sent == 6:
            z2 = torch.cat((hard_orig, hard_aug))
            z0_z2_cos = cls.c_sim(z0.unsqueeze(1), z2.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z0_z2_cos], 1)

        labels = torch.arange(cos_sim.size(0)).long().to(device)
        loss_fct = nn.CrossEntropyLoss()

        if num_sent == 6:
            z2_weight = 0.1
            weights = torch.tensor(
                [
                    [0.0] * (cos_sim.size(-1) - z0_z2_cos.size(-1))
                    + [0.0] * i
                    + [z2_weight]
                    + [0.0] * (z0_z2_cos.size(-1) - i - 1)
                    for i in range(z0_z2_cos.size(-1))
                ]
            ).to(cls.device)
            cos_sim = cos_sim + weights
        c_loss = loss_fct(cos_sim, labels)

    if cls.model_args.a1_loss:
        if num_sent == 6:  # hard orig
            orig_z0 = torch.cat((orig_z0, hard_orig))
            aug_z0 = torch.cat((aug_z0, hard_aug))
        orig_cos_sim = cls.c_sim(orig_z0.unsqueeze(1), orig_z1.unsqueeze(0))
        aug_cos_sim = cls.c_sim(aug_z0.unsqueeze(1), aug_z1.unsqueeze(0))
        a_loss = align_loss_fct(orig_cos_sim, aug_cos_sim)

    if cls.model_args.a2_loss:
        a_loss = align_loss_fct(orig_z0, aug_z0)
        a_loss += align_loss_fct(orig_z1, aug_z1)

    if cls.model_args.a3_loss:
        orig = torch.cat((orig_z0, orig_z1))
        aug = torch.cat((aug_z0, aug_z1))
        ## linear projection step
        proj_orig = linear(orig)
        proj_aug = linear(aug)

        a3_cos_sim = cls.a_sim(proj_orig.unsqueeze(1), proj_aug.unsqueeze(0))
        a3_labels = torch.arange(a3_cos_sim.size(0)).long().to(device)
        a3_loss_fct = nn.CrossEntropyLoss()
        a_loss = a3_loss_fct(a3_cos_sim, a3_labels)

    loss = None
    if a_loss is not None:
        loss = a_loss
    if c_loss is not None:
        loss = c_loss
    if a_loss is not None and c_loss is not None:
        loss = cls.model_args.align_temp * a_loss + c_loss

    if mlm_outputs is not None and mlm_labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        if cls.model_args.a3_loss:
            prediction_scores = cls.lm_head(linear(mlm_outputs.last_hidden_state))
        else:
            prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1)
        )
        if loss is None:
            loss = cls.model_args.mlm_weight * masked_lm_loss
        else:
            loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    return SequenceClassifierOutput(loss=loss, hidden_states=None, attentions=None)


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForMabel(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)
        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)
        self.linear = None
        if self.model_args.a3_loss:
            self.linear = MLPLayer(config, linear=True)

        cl_init(self, config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        bin_mask=None,
    ):
        if sent_emb:
            return sentemb_forward(
                self,
                self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(
                self,
                self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                linear=self.linear,
                bin_mask=bin_mask,
            )


# RoBERTa is not used in this paper
class RobertaForMabel(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)
        cl_init(self, config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        bin_mask=None,
    ):
        if sent_emb:
            return sentemb_forward(
                self,
                self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(
                self,
                self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                bin_mask=bin_mask,
            )
