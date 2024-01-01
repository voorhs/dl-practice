from torch import nn
from src.models import MyResNet, get_seresnext, get_seresnet, get_skresnet, get_seresnet34
import torch
from src.learners import ClfLearner, ClfLearnerConfig


def loaded(model, path):
    learner = ClfLearner.load_from_checkpoint(
        checkpoint_path=path,
        model=model,
        config=ClfLearnerConfig()
    )

    model.load_state_dict(learner.model.state_dict())
    return model


class Ensemble(nn.Module):
    def __init__(self, do_train=True):
        super().__init__()

        models = []
        if do_train:
            models.append(loaded(MyResNet(planes=32), 'weights/myresnet-small.ckpt'))
            models.append(loaded(MyResNet(planes=64), 'weights/myresnet-base.ckpt'))
            models.append(loaded(get_seresnet(), 'weights/seresnet18.ckpt'))
            models.append(loaded(get_seresnext(), 'weights/seresnext.ckpt'))
            models.append(loaded(get_skresnet(), 'weights/skresnet.ckpt'))
            models.append(loaded(get_seresnet34(), 'weights/seresnet34.ckpt'))
        else:
            models.append(MyResNet(planes=32))
            models.append(MyResNet(planes=64))
            models.append(get_seresnet())
            models.append(get_seresnext())
            models.append(get_skresnet())
            models.append(get_seresnet34())


        self.models = nn.ModuleList(models)
        self.models.requires_grad_(False)
        self.clf = nn.Linear(len(models) * 200, 200)
    
    def forward(self, x):
        with torch.no_grad():
            cat_preds = torch.cat([model(x) for model in self.models], dim=1)
        return self.clf(cat_preds)
