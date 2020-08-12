#!/usr/bin/env python
# coding: utf-8

# # HOME: Statistics on HOME Named Entities on German, Latin and Czech - language charters

import argparse
import logging
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

logger = logging.Logger("basic_statistics")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

LANGUAGES = ["czech", "german", "latin"]
NE_TYPES = ["PER", "LOC", "DAT"]
PER, LOC, DAT = "person", "place", "date"
FILENAME_PATTERN = r"page\/.+\.xml$"
PAGE_PATTERN = r"Page$"
TEXTREGION_PATTERN = r"TextRegion$"
TEXTLINE_PATTERN = r"TextLine$"
TEXTEQUIV_PATTERN = r"TextEquiv$"
NAMED_ENTITIES_PATTERN = r"([\w-]+) {offset:(\d+); length:(\d+);( continued)?"

file_pattern = re.compile(FILENAME_PATTERN)
page_pattern = re.compile(PAGE_PATTERN)
textregion_pattern = re.compile(TEXTREGION_PATTERN)
textline_pattern = re.compile(TEXTLINE_PATTERN)
textequiv_pattern = re.compile(TEXTEQUIV_PATTERN)
named_entities_pattern = re.compile(NAMED_ENTITIES_PATTERN)

def get_ne_type(typ, per=PER, loc=LOC, dat=DAT):
    if typ == per:
        return "PER"
    elif typ == loc:
        return "LOC"
    elif typ == dat:
        return "DAT"
    else:
        return

def count_by_type(root_folder, split_lines=True):
    #Init counter df
    index_df = pd.MultiIndex.from_product([NE_TYPES, ["all", "nested"]], names=['type', 'nest'])
    counter_df = pd.DataFrame(np.zeros((6,3)), columns=LANGUAGES, index=index_df, dtype=int)
    counter_df = counter_df.sort_index(level=['type', 'nest'])
    
    #Init stats df
    index_stats_df = pd.MultiIndex.from_product([NE_TYPES, ["avg_char", "avg_tokens"]], names=['type', 'length'])
    stats_df = pd.DataFrame(np.zeros((6,3)), columns=LANGUAGES, index=index_stats_df, dtype=int)
    stats_df = stats_df.sort_index(level=['type', 'length'])

    #Count by language
    for lang in LANGUAGES:
        logger.info("Computing statistics on {} data".format(lang))
        dir_path = os.path.join(root_folder, lang)
        for root, _, files in os.walk(dir_path, topdown=False):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                match = file_pattern.search(file_path)
                if match:
                    tree = ET.parse(file_path).getroot()
                    for child in tree:
                        if not page_pattern.search(child.tag):
                            continue
                        for el in child:
                            if not textregion_pattern.search(el.tag):
                                continue
                            continued = False
                            for line in el:
                                if not textline_pattern.search(line.tag):
                                    continue
                                last_parent_stop = 0
                                #find the child with a text equiv tag
                                for info in line:
                                    if textequiv_pattern.search(info.tag):
                                        text = info[0].text
                                nelist = line.attrib["custom"]
                                entities = named_entities_pattern.findall(nelist)
                                
                                for typ, offset, length, cont in entities:
                                    ne_type = get_ne_type(typ)
                                    if ne_type is None:
                                        continue
                                    offset, length = int(offset), int(length)
                                    end = offset + length
                                    
                                    #handle entities split on two lines => Only update them once
                                    if (not split_lines) and (cont != '') and continued:
                                        continue
                                    continued = (cont != '')
                                    #get nested entities
                                    if offset < last_parent_stop:
                                        counter_df.loc[(ne_type, "nested"), lang] += 1
                                    else:
                                        last_parent_stop = end
                                    counter_df.loc[(ne_type, "all"), lang] += 1
                                    stats_df.loc[(ne_type, "avg_char"), lang] += length
                                    
                                    #Count # of tokens 
                                    len_tokens = len(text[offset:end].split(" "))
                                    stats_df.loc[(ne_type, "avg_tokens"), lang] += len_tokens
    
    #Add average lengths
    counts = counter_df.loc[(slice(None), slice("all")),:].groupby("type")[LANGUAGES].first()
    stats_df.loc[(slice(None), slice("avg_char")),:] = (stats_df.loc[(slice(None), slice("avg_char")),:].groupby("type")[LANGUAGES].first() / counts).values
    stats_df.loc[(slice(None), slice("avg_tokens", "avg_tokens")),:] = (stats_df.loc[(slice(None), slice("avg_tokens", "avg_tokens")),:].groupby("type")[LANGUAGES].first() / counts).values
    
                                    
    return counter_df, stats_df, counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute basic statistics on HOME charters dataset.')
    parser.add_argument('root_folder', type=str, help='path/to/data/root/folder')
    parser.add_argument('--ignore_continue', '-i', action='store_true', default=False,
                        help='If true, tags split between 2 lines are counted twice.')

    args = parser.parse_args()
    home_folder = args.root_folder
    split_lines = args.ignore_continue

    logger.info("Ignoring `continued` : {}".format(split_lines))

    counter_df, stats_df, counts = count_by_type(home_folder, split_lines)
    print("***************COUNTER***********************************")
    print(counter_df.to_markdown())
    print("***************LENGTH STATISTICS*************************")
    print(stats_df.to_markdown())
    