import re

def convert_name_upernet(key):
    key = re.sub("^head/psp/pool([0-9]*)/conv$", lambda m: f"decode_head.psp_modules.{int(m.group(1)) - 1}.1.conv", key)
    key = re.sub("^head/psp/pool([0-9]*)/norm$", lambda m: f"decode_head.psp_modules.{int(m.group(1)) - 1}.1.bn", key)

    key = re.sub("^head/initial4/conv$", lambda m: f"decode_head.bottleneck.conv", key)
    key = re.sub("^head/initial4/norm$", lambda m: f"decode_head.bottleneck.bn", key)
    key = re.sub("^head/initial([0-9]*)/conv$", lambda m: f"decode_head.lateral_convs.{int(m.group(1)) - 1}.conv", key)
    key = re.sub("^head/initial([0-9]*)/norm$", lambda m: f"decode_head.lateral_convs.{int(m.group(1)) - 1}.bn", key)

    key = re.sub("^head/fpn([0-9]*)/conv$", lambda m: f"decode_head.fpn_convs.{int(m.group(1)) - 1}.conv", key)
    key = re.sub("^head/fpn([0-9]*)/norm$", lambda m: f"decode_head.fpn_convs.{int(m.group(1)) - 1}.bn", key)

    key = re.sub("^head/final/conv$", lambda m: f"decode_head.fpn_bottleneck.conv", key)
    key = re.sub("^head/final/norm$", lambda m: f"decode_head.fpn_bottleneck.bn", key)

    key = re.sub("^decode/conv$", lambda m: f"decode_head.conv_seg", key)

    return key
