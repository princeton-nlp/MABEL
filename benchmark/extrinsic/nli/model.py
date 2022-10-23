import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class LinearClassifier(nn.Module):
    def __init__(self, config, args):
        super(LinearClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.config = config
        self.args = args

        self.model = AutoModel.from_pretrained(
            args.model_name_or_path, cache_dir=args.cache_dir, config=config
        )

        classifier_dropout = config.hidden_dropout_prob

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward_eval(self, input_ids, attention_mask, token_type_ids, labels):
        with torch.no_grad():
            return self.forward(
                input_ids, attention_mask, token_type_ids, labels=labels
            )
