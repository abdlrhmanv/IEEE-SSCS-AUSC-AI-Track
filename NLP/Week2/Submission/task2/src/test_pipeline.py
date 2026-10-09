"""Small behavioral checks for leakage prevention, padding and multi-label outputs."""
import unittest
import numpy as np
import torch
from toxic_lstm import (ToxicLSTM, build_vocabulary, clean_text, encode,
                        metric_report, tokenize, tune_thresholds)


class PipelineTests(unittest.TestCase):
    def test_train_vocabulary_does_not_admit_unseen_validation_word(self):
        vocab = build_vocabulary([tokenize(clean_text("I am not angry"))], 100, 1)
        self.assertIn("not", vocab)
        self.assertEqual(encode(["validationonly"], vocab, 10), [1])
        self.assertEqual(encode([], vocab, 10), [1])

    def test_extra_right_padding_cannot_change_prediction(self):
        torch.manual_seed(42)
        model = ToxicLSTM(20, embedding_dim=8, hidden_size=8, dropout=0).eval()
        with torch.no_grad():
            original = model(torch.tensor([[2, 3, 4]]), torch.tensor([3]))
            padded = model(torch.tensor([[2, 3, 4, 0, 0, 0]]), torch.tensor([3]))
        self.assertTrue(torch.allclose(original, padded, atol=1e-6))
        self.assertEqual(tuple(original.shape), (1, 6))

    def test_six_labels_can_be_positive_simultaneously_and_loss_is_finite(self):
        logits = torch.tensor([[10.0] * 6], requires_grad=True)
        target = torch.ones((1, 6))
        loss = torch.nn.BCEWithLogitsLoss()(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(int((torch.sigmoid(logits) > .5).sum()), 6)

    def test_thresholds_recover_separable_low_probability_labels(self):
        y = np.tile([[0], [0], [1], [1]], (1, 6))
        p = np.tile([[.05], [.10], [.30], [.40]], (1, 6))
        thresholds = tune_thresholds(y, p)
        tuned = metric_report(y, p, thresholds)
        self.assertEqual(tuned["macro_f1"], 1.0)
        self.assertLess(metric_report(y, p, .5)["macro_f1"], tuned["macro_f1"])


if __name__ == "__main__":
    unittest.main()
