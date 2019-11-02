import spacy

def extract_objects():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(u"Tashaffi is looking at buying Mona Lisa painting for $1 billion dollars.")
    objects = []

    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
        objects.append(ent.label_)

    return objects