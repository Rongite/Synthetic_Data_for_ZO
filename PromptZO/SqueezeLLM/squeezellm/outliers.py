import torch


@torch.no_grad()
def remove_outliers_by_sensitivity(
    model,
    gradients,
    sensitivity,
):
    module_names = list(model.keys())
    sensitive_weights = dict()
    sensitive_grads = dict()
    total_outliers = 0
    total_weights = 0

    def _body(gweight, weight):
        num_outliers = int(gweight.numel() * sensitivity)
        gweight_sq = gweight.square()
        thres = gweight_sq.reshape(-1).topk(k=num_outliers).values[-1]
        t = gweight_sq > thres

        sensitive_grad = gweight * t
        outlier_weight = weight * t
        weight = weight * ~t
        # print((weight == 0).sum().item() / weight.numel())
        return weight.to(weight.dtype), outlier_weight, sensitive_grad, t.sum().item(), t.numel()

    for _name in module_names:
        weight = model[_name].to(torch.float)
        gweight = gradients[_name].to(torch.float)
        new_weight, outlier_weight, sensitive_grad, _total_outliers, _total_weights = _body(
            gweight, weight
        )
        model[_name] = new_weight
        total_outliers += _total_outliers
        total_weights += _total_weights
        sensitive_weights[_name] = outlier_weight.to_sparse(layout=torch.sparse_csr)
        sensitive_grads[_name] = sensitive_grad.to_sparse(layout=torch.sparse_csr)

    print("p outlier:", total_outliers / total_weights * 100)
    return sensitive_weights, sensitive_grads


@torch.no_grad()
def remove_outliers_by_percentage(
    model,
    outlier_percentage,
):
    module_names = list(model.keys())
    outlier_weights = dict()

    total_outliers = 0
    total_weights = 0

    def _body(weight, outlier_percentage):
        numel_to_exclude = int(model[name].numel() * outlier_percentage)
        t = torch.zeros_like(weight, dtype=torch.bool)
        indices = torch.arange(t.numel(), device=t.device)
        t[
            indices[
                -model[name].view(-1).abs().argsort()[:numel_to_exclude]
            ]
        ] = True
        outlier_weight = weight * t
        weight = weight * ~t
        return weight.to(weight.dtype), outlier_weight, t.sum().item(), t.numel()

    for i, name in enumerate(module_names):
        weight = model[name].to(torch.float)
        new_weight, outlier_weight, _total_outliers, _total_weights = _body(
            weight, outlier_percentage
        )
        model[name] = new_weight
        total_outliers += _total_outliers
        total_weights += _total_weights
        outlier_weights[name] = outlier_weight.to_sparse(layout=torch.sparse_csr)

    print("p outlier:", total_outliers / total_weights * 100)
    return outlier_weights


@torch.no_grad()
def remove_outliers(
    model,
    sensitivity,
    outlier_percentage=0,
    gradients=None,
):
    # model and gradients are dictionary of a layer component
    # where the key is the layer name (e.g. q, k, v) and the value is the weight
    assert isinstance(model, dict)
    assert isinstance(gradients, dict) or gradients is None

    assert outlier_percentage != 0 or sensitivity != 0
    if sensitivity != 0:
        assert gradients is not None

    if sensitivity != 0:
        print("removing outliers by sensitivity")
        sensitive_outlier_weights, sensitive_grads = remove_outliers_by_sensitivity(
            model=model,
            gradients=gradients,
            sensitivity=sensitivity,
        )
    else:
        sensitive_outlier_weights, sensitive_grads = None

    if outlier_percentage != 0:
        print("removing outliers by percentage")
        outlier_weights = remove_outliers_by_percentage(
            model=model,
            outlier_percentage=outlier_percentage,
        )
    else:
        outlier_weights = None

    return sensitive_outlier_weights, sensitive_grads, outlier_weights

