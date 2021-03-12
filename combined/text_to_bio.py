import re
import logging
import argparse

'''
Motivation: Get ref and hyp results of UPV on Home in BIO format to apply our new metric and compare with other results.
The UPV model is a HTR trained to detect beginning and endind of an entity : NER and HTR are done at the same time.
In the prediction file, tags are added between words to indicate begining or ending of entity.
The coherence of the opening and closing of entity tags in the prediction is not assured.

EXAMPLE REF:
(Entities are annoted by line. So if entity overlapps in several lines, re-marked as beginning in next line)
NA-ACK_14060104_01400_r.r1l23 potwrzenye, genz gest <date> dan tento list od narossenye leta Buozyeho tyssyczyeho cztytrczissteho </date>
NA-ACK_14060104_01400_r.r1l24 <date> sessteho, ten pondyeli przyed Buozym krzysstyenii. </date>

Example HYP:
(Here, model failed to predict beginning of date in first line)
NA-ACK_14060104_01400_r.r1l23 potwrzenye, genz gest dan tento list od narossenye leta Buozyeho tyssyczyeho cztytirzissteho </date>
NA-ACK_14060104_01400_r.r1l24 <date> sessteho, ten pondyeli przyed Buozym krzysstyeny. </date>

Metric:
What metric is used for their published score?
- How do they bound detected entities in hypothese if opening/closing tags are not necessary coherent?
- How do they measure perf with nested entities?
What we do:
- Entities selected by line (no reunion when overlapps two lines).
Entity begins at opening tag and ends at corresponding ending tag, or end of line.
If a new entity begins in a other (nested entitie), it is tagged separately and considered as separeted entity.
If first entity continues after nested entities, the rest is tagged as a new entity. <------- This method remains questionnable.


Metric comparison problem raised :
--> Luckily, our metric does not take into account beginning of entities, so if nested, that's ok...
--> Ok for transcription metric, but not Claire classic metric? Check consistency of metrics through different experiments

'''

REF = "./ref_word-test.txt"
HYP = "./hyp_word-test.txt"

ENT_TYPE_MAP = {'persName' : 'PER',
                'placeName' : 'LOC',
                'date' : 'DAT',
                'orgName' : 'ORG'}

PUNCTUATION_MARKS = r"\.,;!?"

def sort_natural(dic) :
    '''
    Takes a dictionary, returns it sorted by key using a natural sort.
    That is, "example12" woudl come after "example2", unlike default python string sort.
    '''
    def text_or_num(text) :
        return int(text) if text.isdigit() else text

    def natural_keys(text) :
        return [text_or_num(c) for c in re.split(r"(\d+)", text)]

    return dict(sorted(dic.items(), key= lambda x : natural_keys(x[0])))


def write_bio_file(file_content, output_name, write_ids=False) :
    ''' Write the extracted word list with BIO labels in a file.

    Input file_content : { doc_id : { line_id : [ (word_1, tag_1), ... ], ... }, ... }
    '''
    # Don't write any file if words is empty

    if file_content:

        all_words = [(line_id, word, tag) for (doc_id, lines) in file_content.items() for line_id, line in lines.items() for word, tag in line]

        if write_ids :
            formatted_words = "\n".join([f"{line_id} {word} {label}" for line_id, word, label in all_words])
        else :
            formatted_words = "\n".join([f"{word} {label}" for _,word,label in all_words])

        with open(output_name, "w") as bio_fd :
            bio_fd.write(formatted_words)


def get_words_units(filename) :
    ''' Get word-tag pair by line by doc in file
        When new doc, aggregate word-tags ordered by line id
        After reaching end of file Don't forget to aggregate last doc:

        Output format : { doc_id : { line_id : [ (word_1, tag_1), ... ], ... }, ... }
    '''

    with open(filename, "r") as ref_fd :

        file_content = dict()
        last_doc_id = ""

        # Browse lines
        for line in ref_fd :

            # Get ids
            words = line.strip().split()
            line_id = words[0]
            doc_id = "".join(line_id.split('.')[:-1])

            # Reform line content
            line = " ".join(words[1:])
            # Add spaces around punctuation
            line = (re.sub(r"\s?([.,:;\?!])\s?", r" \1 ", line)).strip()

            # Process line and tag words in the line
            current_labels = ['O'] # append labels when finding opening, remove when finding ending, &lways tag with last tag of the list
            line_words = [] # [(word, label), (word, label), ...]
            begin = False # track if next word is beginning of entity

            # Browse words in line
            for word in line.split() :

                # Detect entity tag
                label_mo = re.fullmatch(r"</?([^/]+)>", word)

                # If opening tag
                if label_mo and not '/' in label_mo[0] :
                    current_labels.append(ENT_TYPE_MAP[label_mo[1]])
                    begin = True

                # If ending tag
                elif label_mo and '/' in label_mo[0] :
                    try :
                        current_labels.remove(ENT_TYPE_MAP[label_mo[1]])
                    except ValueError :
                        pass

                # If not tag
                else :
                    type = current_labels[-1]
                    full_tag = f"B-{type}" if begin else f"I-{type}"
                    line_words.append((word, full_tag))
                    begin = False

            # First iteration
            if not last_doc_id :
                doc_content = dict()

            # Detect end of doc and sort it's lines
            elif not doc_id == last_doc_id :
                doc_content = sort_natural(doc_content)
                file_content[doc_id] = doc_content
                doc_content = dict()

            doc_content[line_id] = line_words
            last_doc_id = doc_id

        # Process last doc
        doc_content = sort_natural(doc_content)
        file_content[doc_id] = doc_content

    return file_content


def convert_to_bio(filename) :
    file_content = get_words_units(filename)
    output_name = filename.replace('.txt', '.bio')
    write_bio_file(file_content, output_name, write_ids=False)


def main() :
    convert_to_bio(REF)
    convert_to_bio(HYP)


if __name__ == '__main__':
    main()
