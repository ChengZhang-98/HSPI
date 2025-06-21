import re


def get_layer_name(module, layer):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is layer:
            return name
    raise ValueError(f"Cannot find op {layer} in module {module}")


def get_layer_by_name(module, layer_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == layer_name:
            return m
    raise ValueError(f"Cannot find op {layer_name} in module {module}")


def find_matched_pattern(query: str, patterns: list[str]) -> str | None:
    patterns: list[re.Pattern] = [re.compile(pattern) for pattern in patterns]

    matched_patterns = []

    for pattern in patterns:
        if pattern.fullmatch(query):
            matched_patterns.append(pattern)

    if len(matched_patterns) > 1:
        raise ValueError(f"Multiple patterns matched: {matched_patterns}")

    return matched_patterns[0].pattern if len(matched_patterns) == 1 else None


def set_layer_by_name(module, name, new_layer):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = module
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_layer)
    else:
        setattr(module, name, new_layer)
