from transformers import RobertaTokenizer
# tokenization model
PRE_TRAINED_MODEL_NAME = 'roberta-base'

# create the tokenizer to use based on pre trained model
tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)



def convert_to_bert_input_ids(review, max_seq_length, tokenizer=tokenizer):
    encode_plus = tokenizer.encode_plus(
          review,
          add_special_tokens=True,
          ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
          max_length=None, # Replace None
          ### END SOLUTION - DO NOT delete this comment for grading purposes
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
          truncation=True
    )
    return encode_plus['input_ids'].flatten().tolist()

