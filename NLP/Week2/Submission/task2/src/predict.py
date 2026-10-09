"""Predict the six toxicity labels for one comment from a saved checkpoint."""
import argparse
from toxic_lstm import LABELS, predict_texts

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text')
    args=parser.parse_args()
    scores, decisions=predict_texts([args.text])
    print(scores.to_string(index=False))
    print('Predicted labels:',[label for label,flag in zip(LABELS,decisions[0]) if flag])
