from .components import *
from .modules import *
from compressai.models import (
    FactorizedPrior,
    JointAutoregressiveHierarchicalPriors,
    ScaleHyperprior,
)

def load_model(name, task, model_kwargs, optim_kwargs):
    if(name == "vit"):
        model_cls = VisionTransformer
    elif(name == "resnet"):
        model_cls = ResNet
    elif(name == "unet"):
        model_cls = Unet
    elif(name == "factorized_prior"):
        model_cls = FactorizedPrior
    elif(name == "joint_prior"):
        model_cls = JointAutoregressiveHierarchicalPriors
    elif(name == "scale_prior"):
        model_cls = ScaleHyperprior

    model = model_cls(**model_kwargs)

    if(task == "forecasting"):
        module = ForecastLitModule(model, **optim_kwargs)
    elif(task == 'downscaling'):
        module = DownscaleLitModule(model, **optim_kwargs)
    elif(task == 'compression'):
        module = CompressLitModule(model, **optim_kwargs)
    else:
        raise NotImplementedError("Only support foreacasting and downscaling")
    
    return module

def set_climatology(model_module, data_module):
    normalization = data_module.get_out_transforms()
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    model_module.set_denormalization(mean_denorm, std_denorm)
    model_module.set_lat_lon(*data_module.get_lat_lon())
    model_module.set_pred_range(data_module.hparams.pred_range)
    model_module.set_train_climatology(data_module.get_climatology(split = "train"))
    model_module.set_val_climatology(data_module.get_climatology(split = "val"))
    model_module.set_test_climatology(data_module.get_climatology(split = "test"))