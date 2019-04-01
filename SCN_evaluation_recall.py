"""
Semantic Compositional Network https://arxiv.org/pdf/1611.08002.pdf
Developed by Zhe Gan, zg27@duke.edu, July, 12, 2016

Computes the BLEU, ROUGE, METEOR, and CIDER
using the COCO metrics scripts
"""

import argparse
import pickle
import sys

import numpy as np
import json

NOUNS = "nouns"
ADJECTIVES = "adjectives"
OCCURRENCE_DATA = "adjective_noun_occurrence_data"
PAIR_OCCURENCES = "pair_occurrences"
DATA_CAPTIONS = "captions"

RELATION_NOMINAL_SUBJECT = "nsubj"
RELATION_ADJECTIVAL_MODIFIER = "amod"
RELATION_CONJUNCT = "conj"

IMAGES_META_FILENAME = "images_meta.json"

def decode_caption(encoded_caption, word_map):
    rev_word_map = {v: k for k, v in word_map.items()}
    return [rev_word_map[ind] for ind in encoded_caption]

def contains_adjective_noun_pair(pos_tagged_caption, nouns, adjectives):
    noun_is_present = False
    adjective_is_present = False

    for token in pos_tagged_caption.tokens:
        if token.text in nouns:
            noun_is_present = True
        if token.text in adjectives:
            adjective_is_present = True

    dependencies = pos_tagged_caption.dependencies
    caption_adjectives = {
        d[2].text
        for d in dependencies
        if d[1] == RELATION_ADJECTIVAL_MODIFIER and d[0].text in nouns
    } | {
        d[0].text
        for d in dependencies
        if d[1] == RELATION_NOMINAL_SUBJECT and d[2].text in nouns
    }
    conjuncted_caption_adjectives = set()
    for adjective in caption_adjectives:
        conjuncted_caption_adjectives.update(
            {
                d[2].text
                for d in dependencies
                if d[1] == RELATION_CONJUNCT and d[0].text == adjective
            }
            | {
                d[2].text
                for d in dependencies
                if d[1] == RELATION_ADJECTIVAL_MODIFIER and d[0].text == adjective
            }
        )

    caption_adjectives |= conjuncted_caption_adjectives
    combination_is_present = bool(adjectives & caption_adjectives)

    return noun_is_present, adjective_is_present, combination_is_present



def recall_adjective_noun_pairs(
    generated_captions, coco_ids, occurrences_data_file
):
    import stanfordnlp

    # stanfordnlp.download('en', confirm_if_exists=True)
    nlp_pipeline = stanfordnlp.Pipeline()

    with open(occurrences_data_file, "r") as json_file:
        occurrences_data = json.load(json_file)

    nouns = set(occurrences_data[NOUNS])
    adjectives = set(occurrences_data[ADJECTIVES])

    true_positives = np.zeros(5)
    false_negatives = np.zeros(5)
    for coco_id, top_k_captions in zip(coco_ids, generated_captions):
        count = occurrences_data[OCCURRENCE_DATA][coco_id][PAIR_OCCURENCES]

        hit = False
        for caption in top_k_captions:
            pos_tagged_caption = nlp_pipeline(caption).sentences[0]
            _, _, match = contains_adjective_noun_pair(
                pos_tagged_caption, nouns, adjectives
            )
            if match:
                hit = True

        for j in range(count):
            if hit:
                true_positives[j] += 1
            else:
                false_negatives[j] += 1

    recall = true_positives / (true_positives + false_negatives)
    return recall

def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--occurrences-data",
        help="File containing occurrences statistics about adjective noun pairs",
        required=True,
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args

if __name__ == '__main__':
    parsed_args = check_args(sys.argv[1:])

    # generated captions
    generated_captions = pickle.load(open("./decode_results.p", "rb"))
    # generated_captions = {idx: [lines.strip()] for (idx, lines) in enumerate(open('./coco_scn_5k_test.txt', 'rb') )}
    occurrences_data_file = parsed_args.occurrences_data
    coco_ids = [unicode(key) for key in generated_captions.keys()]
    print(recall_adjective_noun_pairs(generated_captions.values(), coco_ids, occurrences_data_file))
    
    
    
    
        

