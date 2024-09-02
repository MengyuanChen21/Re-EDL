import torch.nn as nn


class EnableTestTimeDropout(nn.Module):
    def __init__(self, model):
        super(EnableTestTimeDropout, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        def apply_dropout(m):
            if type(m) == nn.Dropout:
                m.train()

        self.model.apply(apply_dropout)
        return self.model(*args, **kwargs)


def enable_test_time_dropout(model):
    original_mode = model.training
    model.eval()
    dropout_model = EnableTestTimeDropout(model)
    return dropout_model, original_mode
