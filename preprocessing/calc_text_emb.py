"""BLIP-2 text features for the hair-description diffusion conditioning.

Reads per-hairstyle answer files from ``dataset/answers`` and embeds the important +
a few reliable general questions per view, saving the features to
``dataset/features/<idx>/{frontal,back}.pt``.
"""

import argparse
import json
import os
import random
import sys

import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), './submodules/LAVIS'))
from lavis.models import load_model_and_preprocess

sys.path.append(os.getcwd())
from src.utils.text_utils import ALL_QUESTIONS, GENERAL, FRONT, BACK, obtain_blip_features

VIEW_QUESTIONS = {'frontal': FRONT, 'back': BACK}
NUM_GENERAL_SAMPLED = 2


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, _, txt_processors = load_model_and_preprocess(
        name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)

    sub = '001' if args.case == 478 else '000'
    scan_folder = os.path.join(args.dir, f'{args.case:03d}', sub)
    text_path = os.path.join(scan_folder, 'dataset/answers')
    save_path = os.path.join(scan_folder, 'dataset/features')

    # Build embeddings from the per-hairstyle answer files.
    for fname in tqdm(sorted(os.listdir(text_path))):
        idx_name = fname.split('.')[0]
        print(idx_name)
        out_dir = os.path.join(save_path, idx_name)
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(text_path, fname), "r") as f:
            answers = [json.loads(f.readline()) for _ in range(ALL_QUESTIONS)]

        # Per view, keep the important questions plus a few reliable general ones.
        questions = {}
        for view in ('frontal', 'back'):
            unreliable = ['i cannot' in a[f'text_{view}'] for a in answers]
            important = [q for q in VIEW_QUESTIONS[view] if not unreliable[q]]
            general = [q for q in GENERAL if not unreliable[q]]
            questions[view] = sorted(
                important + random.sample(general, min(NUM_GENERAL_SAMPLED, len(general))))

        for view in ('frontal', 'back'):
            embs = [obtain_blip_features(answers[q][f'text_{view}'], model, txt_processors).mean(0)
                    for q in questions[view]]
            torch.save(torch.stack(embs).cpu(), os.path.join(out_dir, f'{view}.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--dir', required=True, type=str)
    parser.add_argument('--case', required=True, type=int)
    args = parser.parse_args()
    main(args)
