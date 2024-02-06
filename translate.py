from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
import torch

import os 
from os.path import join

SEP_TOKEN=2 #specific to nllb I just picked it up

@torch.no_grad()
def _translate_text_chunk(text,tgt_text,tokenizer,model,max_new_tokens=2000,gpu_len=1000):
    gpu_len = min(gpu_len, max_new_tokens)

    # Tokenize and translate the text
    encoded_text = tokenizer(text, return_tensors="pt")
    #manual fix to hf bug 
    encoded_text['input_ids'][:,1]=tokenizer.lang_code_to_id[tokenizer.src_lang]

    tgt_tokens=tokenizer.encode(tgt_text,add_special_tokens=False)
    tgt_tokens=torch.LongTensor([[SEP_TOKEN,tokenizer.lang_code_to_id[tokenizer.tgt_lang]]+tgt_tokens]).to(model.device)

    encoded_text={k:v.to(model.device) for k,v in encoded_text.items()}

    #gpu part
    ans=model.generate(**encoded_text, decoder_input_ids=tgt_tokens,return_dict_in_generate=True,
        penalty_alpha=0.4,max_length=gpu_len)


    #cpu part
    max_new_tokens-=len(ans['sequences'][0][tgt_tokens.shape[1]:])
    if ans['sequences'][0][-1]!=SEP_TOKEN and max_new_tokens:
        print('doing cpu')
        device=model.device
        
        if model.device!='cpu':
            model.to('cpu')
            out['past_key_values']=[[x.cpu() for x in y] for y in out['past_key_values']]

        ans = model.generate(past_key_values=out['past_key_values'],return_dict_in_generate=True,
            max_new_tokens=max_new_tokens,penalty_alpha=0.4)

        model.to(device)

    generated_tokens=ans['sequences']

    # Decode and return the translated text
    return tokenizer.decode(generated_tokens[0][tgt_tokens.shape[1]:], skip_special_tokens=True)

def get_model_and_tokenizer(tgt_lang="heb_Hebr",src_lang="eng_Latn",cuda=True):
    model_name="facebook/nllb-200-3.3B"
    # Initialize the tokenizer and model
    tokenizer = NllbTokenizerFast.from_pretrained(model_name,tgt_lang=tgt_lang,src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    if(cuda):
        model.to('cuda')

    return model,tokenizer

def translate_text(text,tokenizer,model):
    ans=''

    prev=''
    for t in text.split('\n\n'):
        # try:
        prev=_translate_text_chunk(t,prev,tokenizer,model)
        # except torch.cuda.OutOfMemoryError:
        #     model.to('cpu')
        #     prev=_translate_text_chunk(t,prev,tokenizer,model)
        #     model.to('cuda')
        ans+='\n\n'+prev
    return ans

if __name__=="__main__":
    import sqlite3
    #s="תופעת_רשת"

    path=join('data','wikisql.db')

    conn = sqlite3.connect(path)  # Replace 'your_database.db' with your database file path
    cursor = conn.cursor()

    # Execute a query to select English texts
    #cursor.execute("SELECT text FROM texts WHERE id IN (SELECT id FROM main_data WHERE lang = 'en') LIMIT 5")

    cursor.execute("""
        SELECT texts.text
        FROM main_data
        INNER JOIN texts ON main_data.id = texts.id
        WHERE main_data.lang = 'en'
        LIMIT 5
    """)


    # Fetch the results
    english_texts = [x[0] for x in cursor.fetchall()]

    # Close the database connection
    conn.close()

    # # Print the retrieved English texts
    # for text in english_texts:
    #     print(text)  
    text=english_texts[0]
    
    model,tokenizer=get_model_and_tokenizer()
    print(text)
    print(10*'\n')
    trans=translate_text(text,tokenizer,model)
    print(trans)
    