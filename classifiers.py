#%%
from convert_example_to_features import *
from transformers import BertModel, BertPreTrainedModel, BertTokenizer
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from config import *
from transformers.activations import gelu
import math
class SoftAttentionLayer(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size,hidden_size)
        self.tanh = nn.Tanh()
        self.query = nn.Linear(hidden_size,1,bias=False)
        self.softmax = nn.Softmax(dim=1)
        #dimension 1 means along the rows do softmax (vertical)

    def forward(self, hidden_states):
        # hidden_state size should be batch_size * sequence length * hidden_size
        input_tensor = hidden_states
        hidden_states = self.dense(hidden_states)
        hidden_states = self.tanh(hidden_states)
        hidden_states = self.query(hidden_states)
        hidden_states = self.softmax(hidden_states) #output shape is [10,512,1]
        hidden_states = hidden_states * input_tensor
        hidden_states = torch.sum(hidden_states,1)
        return hidden_states

class SelfAttention(nn.Module):
    def __init__(self, config,hidden_size):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.hidden_size = hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size,hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_mask = encoder_attention_mask
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class SelfOutput(nn.Module):
    def __init__(self, config,hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size,hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Intermediate(nn.Module):
    def __init__(self, config,hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = gelu
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class Attention(nn.Module):
    def __init__(self, config,hidden_size):
        super().__init__()
        self.self = SelfAttention(config,hidden_size)
        self.output = SelfOutput(config,hidden_size)


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class Output(nn.Module):
    def __init__(self, config,hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Layer(nn.Module):
    def __init__(self, config,hidden_size):
        super().__init__()
        self.Attention = Attention(config,hidden_size)
        self.intermediate = Intermediate(config,hidden_size)
        self.output = Output(config,hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.Attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs # outputs is a tuple, we want the first element in the tuple

class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            reshaped_logits = logits.view(-1, 5) 
            _, reshaped_labels = torch.max(labels.view(-1, 5), 1)
            
            loss_fct = CrossEntropyLoss()

            loss = loss_fct(reshaped_logits, reshaped_labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class Baseline(BertPreTrainedModel):
    
    def __init__(self, config):
        super(Baseline, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            reshaped_logits = logits.view(-1, 5) 
            _, reshaped_labels = torch.max(labels.view(-1, 5), 1)
            
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, reshaped_labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class LEHS(BertPreTrainedModel):
            

    def __init__(self, config):
        super(LEHS, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attentionLayer_input = Layer(config,config.hidden_size)
        self.attentionLayer_lower = Layer(config,config.hidden_size)
        self.attentionLayer_upper = Layer(config,config.hidden_size)

        self.LSTM1 = nn.LSTM(input_size = config.hidden_size,hidden_size = 200,num_layers = 1,bidirectional = True)
        self.LSTM2 = nn.LSTM(input_size = config.hidden_size,hidden_size = 200,num_layers = 1,bidirectional = True) # the output dimension of LSTM1 is seq_len * batch * (hidden size * 2)  because it is doubled
        self.softAttention_lower = SoftAttentionLayer(400)
        self.softAttention_upper = SoftAttentionLayer(400)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(800,800)
        self.fc2 = nn.Linear(800,1)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None,tokenizer=None):
                
        def Label_attention(query_vector, sequence_vector):
            # the query_vector should be batch_size * hidden_size
            # sequence_vector : batch * sequence length * hidden size
            device = query_vector.device
            batch_size = query_vector.shape[0]
            hidden_size = query_vector.shape[1]
            query_vector = query_vector.unsqueeze(1) # batch * 1 * hidden_size
            temp = (torch.matmul(query_vector,sequence_vector.transpose(1,2))/torch.sqrt(torch.tensor(hidden_size,dtype = float,device = device))).squeeze(1) #batch * seq_len
            label_attention_weight = self.softmax(temp) #batch * seq_len
            label_attention_vector = torch.matmul(label_attention_weight.unsqueeze(1),sequence_vector) #batch * 1 * hidden_size
            return label_attention_vector #batch * 1 * hidden_size

        def Hierarchical_attention(label_attention_vector, RNN_outputs):
            # the label_attention_vector should be batch_size * 1 * hidden_size
            # the RNN_outputs should be batch_size * seq_len * hidden_size
            device = label_attention_vector.device
            hierarchical_attention_weight = torch.matmul(label_attention_vector,RNN_outputs.transpose(1,2)) #batch*1*seq_len
            hierarchical_attention_vector = torch.matmul(hierarchical_attention_weight,RNN_outputs) #batch_size * 1 * hidden_size
            hierarchical_attention_vector = hierarchical_attention_vector.squeeze(1) #batch_size  * hidden_size
            return hierarchical_attention_vector #batch_size  * hidden_size

        self.bert.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logits_score = torch.tensor([],dtype = torch.float32).to(device)
        tokenizer = BertTokenizer.from_pretrained("savedModel/bert-base-chinese-vocab.txt")

        for label in all_label:
            LowerLabelds = label_2_ds_dict[label]
            UpperLabel = lower_2_upper_label_dict[label]
            UpperLabel = upper_ds_dict[UpperLabel]
            
            lower_label_input_ids,lower_label_input_mask,lower_label_segment_ids = convert_label_description_to_features(LowerLabelds,512,tokenizer)
            upper_label_input_ids,upper_label_input_mask,upper_label_segment_ids = convert_label_description_to_features(UpperLabel,512,tokenizer)

            batch_size = input_ids.shape[0]
            #create batch size * hidden size of the label description vector
            lower_label_input_ids = lower_label_input_ids.repeat(batch_size,1).to(device)
            lower_label_input_mask = lower_label_input_mask.repeat(batch_size,1).to(device)
            lower_label_segment_ids = lower_label_segment_ids.repeat(batch_size,1).to(device)
            upper_label_input_ids = upper_label_input_ids.repeat(batch_size,1).to(device)
            upper_label_input_mask = upper_label_input_mask.repeat(batch_size,1).to(device)
            upper_label_segment_ids = upper_label_segment_ids.repeat(batch_size,1).to(device)

            # -------------------------------BERT--------------------------------------------------------------
            input_text_outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                attention_mask=attention_mask, head_mask=head_mask)

            lower_text_outputs = self.bert(lower_label_input_ids, token_type_ids=lower_label_segment_ids,
                                attention_mask=lower_label_input_mask)

            upper_text_outputs = self.bert(upper_label_input_ids, token_type_ids=upper_label_segment_ids,
                                attention_mask=upper_label_input_mask)

            output_text_sequence = input_text_outputs[0] #batch * seq_len * hidden size
            output_lower_sequence = lower_text_outputs[0] #batch * seq_len * hidden size
            output_upper_sequence = upper_text_outputs[0] #batch * seq_len * hidden size
            
            # # -------------------------------Self Attention---------------------------------------
            # output_text_sequence = self.dropout(output_text_sequence)
            # output_lower_sequence = self.dropout(output_lower_sequence)
            # output_upper_sequence = self.dropout(output_upper_sequence)

            # output_text_sequence = self.attentionLayer_input(output_text_sequence)
            # output_lower_sequence = self.attentionLayer_lower(output_lower_sequence)
            # output_upper_sequence = self.attentionLayer_upper(output_upper_sequence)

            # -------------------------------LSTM Encoder----------------------------------------
            input_of_LSTM = output_text_sequence.transpose(0,1) #seq_len * batch * hidden size
            input_lower_of_LSTM = output_lower_sequence.transpose(0,1) #seq_len * batch * hidden size
            input_upper_of_LSTM = output_upper_sequence.transpose(0,1) #seq_len * batch * hidden size

            input_of_LSTM = self.dropout(input_of_LSTM)
            out,(hidden,cn) = self.LSTM1(input_of_LSTM) #out dimension is [seq_len,batch,hidden size*2]

            input_lower_of_LSTM = self.dropout(input_lower_of_LSTM)
            out_lower,(hidden_lower,cn_lower) = self.LSTM2(input_lower_of_LSTM) #out dimension is [seq_len,batch,hidden size*2]

            input_upper_of_LSTM = self.dropout(input_upper_of_LSTM)
            out_upper,(hidden_upper,cn_upper) = self.LSTM2(input_upper_of_LSTM) #out dimension is [seq_len,batch,hidden size*2]

            out = out.transpose(0,1) # batch * sequence length * 2hidden size
            out_lower = out_lower.transpose(0,1) # batch * sequence length * 2hidden size
            out_upper = out_upper.transpose(0,1) # batch * sequence length * 2hidden size

            # -------------------------------Soft Attention part---------------------------------
            out_lower_softattention_vector = self.softAttention_lower(out_lower)    #batch_size * 400
            out_upper_softattention_vector = self.softAttention_upper(out_upper)    #batch_size * 400

            # -------------------------------Cross Label Attention part--------------------------

            upper_label_attention_vector = Label_attention(out_upper_softattention_vector,out_lower)
            lower_label_attention_vector = Label_attention(out_lower_softattention_vector,out_upper)
            
            # --------------------------------Hierarchical Attention part--------------------------
            upper_hirarchi_attention_vector = Hierarchical_attention(upper_label_attention_vector,out) #batch_size  * hidden_size
            lower_hirarchi_attention_vector = Hierarchical_attention(lower_label_attention_vector,out) #batch_size  * hidden_size

            # -------------------------------Concat Vector -------------------------------------------
            concat_vector = torch.cat((upper_hirarchi_attention_vector,lower_hirarchi_attention_vector),dim= 1)

            # -------------------------------Matcher----------------------------------------------
            concat_vector = self.dropout(concat_vector)
            concat_vector = self.relu(concat_vector)
            fc1_out = self.fc1(concat_vector)
            fc1_out = self.dropout(fc1_out)
            fc1_out = self.relu(fc1_out)
            logits = self.fc2(fc1_out)
            logits_score = torch.cat((logits_score,logits),dim = 1)

        logits = logits_score
        outputs = (logits,) + input_text_outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                label_cuda = torch.tensor(labels,device="cuda")
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1),label_cuda.view(-1))
            else:
                label_cuda = torch.tensor(labels,device="cuda")
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1,self.num_labels),label_cuda.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

CLASSIFIER_CLASSES = {
    'default': BertForSequenceClassification,
    'baseLine': Baseline,
    'LEHS': LEHS
}

# %%
