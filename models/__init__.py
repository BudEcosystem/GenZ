from . import causal


MODEL_REGISTRY = {
    "causal": causal.Causal,
}

def get_model(model_args):
    return MODEL_REGISTRY[model_args.model_type](model_args)