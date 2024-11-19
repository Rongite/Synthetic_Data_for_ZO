from transformers import AutoModelForCausalLM, AutoModelForMaskedLM


def load_model(model, cache_dir=None):
    if 'roberta' in model:
        model = AutoModelForMaskedLM.from_pretrained(
            model, torch_dtype="auto", cache_dir=cache_dir
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype="auto", cache_dir=cache_dir
        )
    return model


def parse_model(model):
    if "opt" in str(type(model)).lower():
        model_type = "opt"
    elif "mistral" in str(type(model)).lower():
        model_type = "mistral"
    elif 'roberta' in str(type(model)).lower():
        model_type = "roberta"
    else:
        # additional rules should be added to support other models
        model_type = "llama"
    print(f"Model type : {model_type}")

    return model_type


def get_module_names(model_type):
    if model_type == "opt" or model_type == 'roberta':
        return ["q", "k", "v", "o", "up", "down"]
    else:
        assert model_type == "llama" or model_type == "mistral"
        return ["q", "k", "v", "o", "gate", "up", "down"]


def get_modules(layer, model_type):
    if model_type == "opt":
        modules = [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.out_proj,
            layer.fc1,
            layer.fc2,
        ]
    elif model_type == 'roberta':
        modules = [
            layer.attention.self.query,
            layer.attention.self.key,
            layer.attention.self.value,
            layer.attention.output.dense,
            layer.intermediate.dense,
            layer.output.dense,
        ]
    else:
        # llama or vicuna
        assert model_type == "llama" or model_type == "mistral"
        modules = [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        ]

    return modules


def get_sequential(model_type):
    if model_type == "opt":
        return [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "fc1",
            "fc2",
        ]
    elif model_type == 'roberta':
        return [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
            "intermediate.dense",
            "output.dense",
        ]
    else:
        assert model_type == "llama" or model_type == "mistral"
        return [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]


def get_model(model, model_type):
    if model_type == "opt":
        return model.model.decoder
    elif model_type == "roberta":
        return model.roberta
    else:
        assert model_type == "llama" or model_type == "mistral"
        return model.model


def get_layers(model, model_type):
    _model = get_model(model, model_type)
    if model_type == "opt":
        return _model.layers
    elif model_type == 'roberta':
        return _model.encoder.layer
    else:
        assert model_type == "llama" or model_type == "mistral"
        return _model.layers


def get_layers_name(model_type):
    if model_type == "opt":
        return "model.decoder.layers"
    else:
        assert model_type == "llama" or model_type == "mistral"
        return "model.layers"


def get_embedding(model, model_type):
    _model = get_model(model, model_type)
    if model_type == "opt":
        return [_model.embed_tokens, _model.embed_positions]
    else:
        assert model_type == "llama" or model_type == "mistral"
        return [_model.embed_tokens]


def get_norm(model, model_type):
    _model = get_model(model, model_type)
    if model_type == "opt":
        return _model.final_layer_norm
    else:
        assert model_type == "llama" or model_type == "mistral"
        return _model.norm
