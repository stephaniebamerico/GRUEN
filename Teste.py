import difflib
import editdistance
import math
import numpy as np
import re
import spacy
import string
import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, BertForMaskedLM
from transformers import glue_convert_examples_to_features
from transformers.data.processors.utils import InputExample
from wmd import WMD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


""" Processing """
def preprocess_candidates(candidates):
    for i in range(len(candidates)):
        candidates[i] = candidates[i].strip()
        candidates[i] = '. '.join(candidates[i].split('\n\n'))
        candidates[i] = '. '.join(candidates[i].split('\n'))
        candidates[i] = '.'.join(candidates[i].split('..'))
        candidates[i] = '. '.join(candidates[i].split('.'))
        candidates[i] = '. '.join(candidates[i].split('. . '))
        candidates[i] = '. '.join(candidates[i].split('.  . '))
        while len(candidates[i].split('  ')) > 1:
            candidates[i] = ' '.join(candidates[i].split('  '))
        myre = re.search(r'(\d+)\. (\d+)', candidates[i])
        while myre:
            candidates[i] = 'UNK'.join(candidates[i].split(myre.group()))
            myre = re.search(r'(\d+)\. (\d+)', candidates[i])
        candidates[i] = candidates[i].strip()
    processed_candidates = []
    for candidate_i in candidates:
        sentences = sent_tokenize(candidate_i)
        out_i = []
        for sentence_i in sentences:
            if len(sentence_i.translate(str.maketrans('', '', string.punctuation)).split()) > 1:  # More than one word.
                out_i.append(sentence_i)
        processed_candidates.append(out_i)
    return processed_candidates


""" Scores Calculation """
def get_lm_score(sentences):
    def score_sentence(sentence, tokenizer, model):
        # if len(sentence.strip().split()) <= 1:
        #     return 10000
        tokenize_input = tokenizer.tokenize(sentence)
        if len(tokenize_input) > 510:
            tokenize_input = tokenize_input[:510]
        input_ids = torch.tensor(tokenizer.encode(tokenize_input)).unsqueeze(0).to(device)
        with torch.no_grad():
            loss = model(input_ids, labels=input_ids)[0]
        return math.exp(loss.item())

    model_name = 'bert-base-cased'
    model = BertForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(model_name)
    lm_score = []
    for sentence in tqdm(sentences):
        if len(sentence) == 0:
            lm_score.append(0.0)
            continue
        score_i = 0.0
        for x in sentence:
            score_i += score_sentence(x, tokenizer, model)
        score_i /= len(sentence)
        lm_score.append(score_i)
    return lm_score


def get_cola_score(sentences):
    def load_pretrained_cola_model(model_name, saved_pretrained_CoLA_model_dir):
        config_class, model_class, tokenizer_class = (BertConfig, BertForSequenceClassification, BertTokenizer)
        config = config_class.from_pretrained(saved_pretrained_CoLA_model_dir, num_labels=2, finetuning_task='CoLA')
        tokenizer = tokenizer_class.from_pretrained(saved_pretrained_CoLA_model_dir, do_lower_case=0)
        model = model_class.from_pretrained(saved_pretrained_CoLA_model_dir, from_tf=bool('.ckpt' in model_name), config=config).to(device)
        model.eval()
        return tokenizer, model

    def evaluate_cola(model, candidates, tokenizer, model_name):

        def load_and_cache_examples(candidates, tokenizer):
            max_length = 128
            examples = [InputExample(guid=str(i), text_a=x) for i,x in enumerate(candidates)]
            features = glue_convert_examples_to_features(examples, tokenizer, label_list=["0", "1"], max_length=max_length, output_mode="classification")
            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_labels = torch.tensor([0 for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([[0.0]*max_length for f in features], dtype=torch.long)
            dataset = torch.utils.data.TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
            return dataset

        eval_dataset = load_and_cache_examples(candidates, tokenizer)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, sampler=torch.utils.data.SequentialSampler(eval_dataset), batch_size=max(1, torch.cuda.device_count()))
        preds = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3]}
                if model_name.split('-')[0] != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if model_name.split('-')[0] in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        return preds[:, 1].tolist()

    def convert_sentence_score_to_paragraph_score(sentence_score, sent_length):
        paragraph_score = []
        pointer = 0
        for i in sent_length:
            if i == 0:
                paragraph_score.append(0.0)
                continue
            temp_a = sentence_score[pointer:pointer + i]
            paragraph_score.append(sum(temp_a) / len(temp_a))
            pointer += i
        return paragraph_score

    model_name = 'bert-base-cased'
    saved_pretrained_CoLA_model_dir = './cola_model/' + model_name + '/'
    tokenizer, model = load_pretrained_cola_model(model_name, saved_pretrained_CoLA_model_dir)
    candidates = [y for x in sentences for y in x]
    sent_length = [len(x) for x in sentences]
    cola_score = evaluate_cola(model, candidates, tokenizer, model_name)
    cola_score = convert_sentence_score_to_paragraph_score(cola_score, sent_length)
    return cola_score


def get_grammaticality_score(processed_candidates):
    lm_score = get_lm_score(processed_candidates)
    cola_score = get_cola_score(processed_candidates)
    grammaticality_score = [1.0 * math.exp(-0.5*x) + 1.0 * y for x, y in zip(lm_score, cola_score)]
    grammaticality_score = [max(0, x / 8.0 + 0.5) for x in grammaticality_score]  # re-scale
    return grammaticality_score


def get_redundancy_score(all_summary):
    def if_two_sentence_redundant(a, b):
        """ Determine whether there is redundancy between two sentences. """
        if a == b:
            return 4
        if (a in b) or (b in a):
            return 4
        flag_num = 0
        a_split = a.split()
        b_split = b.split()
        if max(len(a_split), len(b_split)) >= 5:
            longest_common_substring = difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
            LCS_string_length = longest_common_substring.size
            if LCS_string_length > 0.8 * min(len(a), len(b)):
                flag_num += 1
            LCS_word_length = len(a[longest_common_substring[0]: (longest_common_substring[0]+LCS_string_length)].strip().split())
            if LCS_word_length > 0.8 * min(len(a_split), len(b_split)):
                flag_num += 1
            edit_distance = editdistance.eval(a, b)
            if edit_distance < 0.6 * max(len(a), len(b)):  # Number of modifications from the longer sentence is too small.
                flag_num += 1
            number_of_common_word = len([x for x in a_split if x in b_split])
            if number_of_common_word > 0.8 * min(len(a_split), len(b_split)):
                flag_num += 1
        return flag_num

    redundancy_score = [0.0 for x in range(len(all_summary))]
    for i in range(len(all_summary)):
        flag = 0
        summary = all_summary[i]
        if len(summary) == 1:
            continue
        for j in range(len(summary) - 1):  # for pairwise redundancy
            for k in range(j + 1, len(summary)):
                flag += if_two_sentence_redundant(summary[j].strip(), summary[k].strip())
        redundancy_score[i] += -0.1 * flag
    return redundancy_score


def get_focus_score(all_summary):
    def compute_sentence_similarity():
        nlp = spacy.load('en_core_web_md')
        nlp.add_pipe(WMD.SpacySimilarityHook(nlp), last=True)
        all_score = []
        for i in range(len(all_summary)):
            if len(all_summary[i]) == 1:
                all_score.append([1.0])
                continue
            score = []
            for j in range(1, len(all_summary[i])):
                doc1 = nlp(all_summary[i][j-1])
                doc2 = nlp(all_summary[i][j])
                try:
                    score.append(1.0/(1.0 + math.exp(-doc1.similarity(doc2)+7)))
                except:
                    score.append(1.0)
            all_score.append(score)
        return all_score

    all_score = compute_sentence_similarity()
    focus_score = [0.0 for x in range(len(all_summary))]
    for i in range(len(all_score)):
        if len(all_score[i]) == 0:
            continue
        if min(all_score[i]) < 0.05:
            focus_score[i] -= 0.1
    return focus_score


def get_gruen(candidates):
    processed_candidates = preprocess_candidates(candidates)
    grammaticality_score = get_grammaticality_score(processed_candidates)
    redundancy_score = get_redundancy_score(processed_candidates)
    focus_score = get_focus_score(processed_candidates)
    gruen_score = [min(1, max(0, sum(i))) for i in zip(grammaticality_score, redundancy_score, focus_score)]
    return gruen_score


if __name__ == "__main__":
    # candidates = ["This is a good example.",
    #               "This is a bad example. It is ungrammatical and redundant. Orellana shown red card for throwing grass at Sergio Busquets. Orellana shown red card for throwing grass at Sergio Busquets.",
    #               "Hello, doctor.",
    #               "I like apples. I really like apples.",
    #               "There was a boy and a girl, they were siblings. He smilling at her. She was angry.",
    #               "Once upon a time."]
    text = "King Bluebeard Next to a great forest there lived an old man who had three sons and two daughters. Once they were sitting together thinking of nothing when a splendid carriage suddenly drove up and stopped in front of their house. A dignified gentleman climbed from the carriage, entered the house, and engaged the father and his daughters in conversation. Because he especially liked the youngest one, he asked the father if he would not give her to him to be his wife. This seemed to the father to be a good marriage, and he had long desired to see his daughters taken care of while he was still alive. However, the daughter could not bring herself to say yes, for the strange knight had an entirely blue beard, which caused her to shudder with fear whenever she looked at him. She went to her brothers, who were valiant knights, and asked them for advice. The brothers thought that she should accept Bluebeard, and they gave her a little whistle, saying, \"If you are ever threatened, just blow this whistle, and we will come to your aid!\" Thus she let herself be talked into becoming the strange man's wife, but she did arrange for her sister to accompany her when King Bluebeard took her to his castle. When the young wife arrived there, there was great joy throughout the entire castle, and King Bluebeard was very happy as well. This continued for about four weeks, and then he said that he was going on a journey. He turned all the keys of the castle over to his wife, saying, \"You may go anywhere in the castle, unlock everything, and look at anything you want to, except for one door, to which this little golden key belongs. If you value your life, you are not allowed to open it!\" \"Oh no!\" she said, adding that she surely would not open that door. But after the king had been away for a while, she could find no rest for constantly thinking about what there might be in the forbidden chamber. She was just about to unlock it when her sister approached her and held her back. However, on the morning of the fourth day, she could no longer resist the temptation, and taking the key she secretly crept to the room, stuck the key into the lock, and opened the door. Horrified, she saw that the entire room was filled with corpses, all of them women. She wanted to slam the door shut immediately, but the key fell out and into the blood. She quickly picked it up, but it was stained with blood. And however much she rubbed and cleaned it, the stains would not go away. With fear and trembling she went to her sister. When King Bluebeard finally returned from his journey, he immediately asked for the golden key. Seeing the bloodstains on it, he said, \"Wife, why did you not heed my warning? Your hour has now struck! Prepare yourself to die, for you have been in the forbidden room!\" Crying, she went to her sister, who lived upstairs in the castle. While she was bemoaning her fate to her, the sister thought of the whistle that she had received from her brothers, and said, \"Give me the whistle! I shall send a signal to our brothers. Perhaps they will be able to help!\" And she blew the whistle three times, issuing a bright sound that rang through the woods. An hour later they heard Bluebeard rustling up the stairs to get his wife and slaughter her. \"Oh God, oh God!\" she cried out. \"Aren't my brothers coming?\" She rushed to the door and locked it, then fearfully stood there holding it shut as well. Bluebeard pounded on the door, crying out that she should open it, and when she did not do so, he tried to break it down. \"Oh sister, oh sister, aren't my brothers coming?\" she said to her sister, who was standing at the window looking out into the distance. She replied, \"I don't see anyone yet. \" Meanwhile, Bluebeard was breaking the door apart more and more, and the opening was almost large enough for him to get through, when three knights suddenly appeared before the castle. The sister cried from the window as loudly as she could, \"Help! Help!\" and waved to her brothers. They stormed up the stairs to where they had heard their sister's cry for help. There they saw King Bluebeard, sword in hand, standing before the broken door, and they heard their sister screaming inside the room. Immediately sensing what he was up to, they quickly ran their daggers into his breast and killed him. When the brothers learned what the godless king was going to do to their sister, and that he had already killed so many women, they destroyed his castle, so that there was not one stone remaining on another one. They took with them all his treasures, and lived happily with their sisters in their father's house."
    hi = len(text) 
    lo = 0
    while (hi > lo):
        mid = int((hi + lo)/2)

        candidates = [text[lo:mid], text[mid:hi], text[lo:hi]]
        print("--------------")
        for candi in candidates:
            print()
            print (candi)
            print()
        # print(candidates)
        print("--------------")
    #for idx, val in enumerate(candidates):
    #    f = open ("teste" + str(idx), "w")
    #    f.write(val)
    #    f.close()
        gruen_score = get_gruen(candidates)
        # print(gruen_score)
        # float = 2.154327
        # format_float = "{:.2f}".format(float)
        format_float = "{:.15f}".format(gruen_score[0])
        print(format_float)

        format_float = "{:.15f}".format(gruen_score[1])
        print(format_float)
        format_float = "{:.15f}".format(gruen_score[2])
        print(format_float)
        if (gruen_score[0] == 0):
            hi = mid
        if (gruen_score[1] == 0):
            lo = mid+1
    print (hi, lo)

