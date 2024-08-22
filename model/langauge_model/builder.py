from models.language_models.vicuna import Vicuna

def llm_builder(config):
    """
    arg: configuration
    return: constructed llm model
    """

    model_cls = config.llm_config.model_cls
    device = config.device

    if model_cls == 'vicuna-7b-v1.5':
        return SoccerVicuna(config.llm_config, device)

    raise ValueError(f'invalid llm type: {model_cls}')