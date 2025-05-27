#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 19:39:20 2022

@author: saurabh
"""

import sys
sys.path.append("local/")
from jitsi_wer import wer
import os
import argparse

def get_args():
    """ Get args from stdin."""
    parser = argparse.ArgumentParser(
        description="""Input Parameters""")

    parser.add_argument("--ref_path", type=str,
            default="text_ref", help="Path of reference text file")
    parser.add_argument("--hyp_path", type=str,
            default="utt_hyp_per_utt_eval_v5", help="Path of hypothesis text file")
    parser.add_argument("--output_path", type=str,
            default="output.txt", help="Path of the output text file")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    ref = args.ref_path
    hyp = args.hyp_path
    output_path = args.output_path
    
    pathr0 = ref
    pathr1 = hyp
    
    utt2text0 = {}
    utt2text1 = {}
    uttids0 = []
    uttids1 = []
    
    with open(pathr0, encoding='utf-8', errors='ignore') as f0, open(pathr1, encoding='utf-8', errors='ignore') as f1:
        for x in f0:
            utt2text0[x.split()[0]] = ' '.join(x.split()[1:])
            uttids0.append(x.split()[0])
        for x in f1:
            utt2text1[x.split()[0]] = ' '.join(x.split()[1:])
            uttids1.append(x.split()[0])
    
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for key, value in utt2text0.items():
            wer_dict = wer(utt2text0[key], utt2text1[key], return_dict=True)
            output_line = f"{key} #csid {wer_dict['hits']} {wer_dict['substitutions']} {wer_dict['insertions']} {wer_dict['deletions']}\n"
            output_file.write(output_line)
    
if __name__ == '__main__':
    main()
