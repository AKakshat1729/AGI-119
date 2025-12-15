# nlu/nlu_live.py
import nltk

nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("maxent_ne_chunker_tab", quiet=True)
nltk.download("words", quiet=True)

def get_entities(text):
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    tree = nltk.ne_chunk(tags)
    entities = []
    for subtree in tree:
        if isinstance(subtree, nltk.Tree):
            entity = " ".join([word for word, tag in subtree.leaves()])
            label = subtree.label()
            entities.append({"entity": entity, "type": label})
    return entities

def get_roles(text):
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    roles = []
    for w, t in tags:
        if t.startswith("NN"): roles.append({"word": w, "role": "entity"})
        elif t.startswith("VB"): roles.append({"word": w, "role": "action"})
    return roles

def nlu_process(text, tone_obj):
    return {
        "transcript": text,
        "sentiment": tone_obj["sentiment"],
        "emotions": tone_obj["emotions"],
        "entities": get_entities(text),
        "semantic_roles": get_roles(text)
    }
