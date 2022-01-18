import tensorflow as tf
from ... import hrnet, decode, ocr, resnet, psa, mscale
from ...util import *
import tfcv, os, requests, re
from functools import partial

from ..hrnet.util import preprocess, config, convert_name_hrnet, convert_name_ocr

def download_cmu_box(url, name):
    file = os.path.join(os.path.expanduser("~"), ".keras", "datasets", name)
    if not os.path.isfile(file):
        r = requests.get(url)

        match = re.search("\"itemID\":([0-9]+)", r.text)
        item_id = int(match.group(1))

        match = re.search("\"requestToken\":\"(.+?)\"", r.text)
        request_token = match.group(1)

        data = {"fileIDs": [f"file_{item_id}"]}
        headers = {
            "Request-Token": request_token,
            "X-Request-Token": request_token,
        }
        r = requests.post("https://cmu.app.box.com/app-api/enduserapp/elements/tokens", json=data, headers=headers, cookies=r.cookies)
        match = re.search("\"read\":\"(.+?)\"", r.text)
        auth = match.group(1)

        headers = {
            "Authorization": f"Bearer {auth}",
            "BoxApi": f"shared_link={url}",
        }
        r = requests.get(f"https://api.box.com/2.0/files/{item_id}?fields=download_url", headers=headers, cookies=r.cookies)
        download_url = r.json()["download_url"]

        assert file == tf.keras.utils.get_file(name, download_url)
    return file

def convert_name(name):
    if "stem" in name or "block" in name:
        if "/psa/" in name:
            def sub_inner(name):
                name = re.sub("^spatial/query/conv$", lambda m: "conv_q_right", name)
                name = re.sub("^spatial/value/conv$", lambda m: "conv_v_right", name)
                name = re.sub("^channel/query/conv$", lambda m: "conv_q_left", name)
                name = re.sub("^channel/value/conv$", lambda m: "conv_v_left", name)

                name = re.sub("^spatial/weight/conv$", lambda m: "conv_up", name)

                name = re.sub("^spatial/weight/1/conv$", lambda m: "conv_up.0", name)
                name = re.sub("^spatial/weight/1/norm$", lambda m: "conv_up.1", name)
                name = re.sub("^spatial/weight/2/conv$", lambda m: "conv_up.3", name)

                return name

            def sub(m):
                return f"module.backbone.stage{m.group(1)}.{int(m.group(2)) - 1}.branches.{int(m.group(3)) - 1}.{int(m.group(4)) - 1}.deattn." + sub_inner(m.group(5))
            name = re.sub("^block([0-9]*)/module([0-9]*)/branch([0-9]*)/unit([0-9]*)/1/psa/(.*)$", sub, name)
        else:
            name = "module.backbone." + convert_name_hrnet(name)
    elif "ocr" in name or name.startswith("final"):
        name = "module.ocr." + convert_name_ocr(name)
        name = name.replace("module.ocr.aux_head.1", "module.ocr.aux_head.1.0")
        name = name.replace("module.ocr.aux_head.3", "module.ocr.aux_head.2")
        name = name.replace("module.ocr.conv3x3_ocr.1", "module.ocr.conv3x3_ocr.1.0")
    else:
        name = name.replace("norm", "bn")
        name = re.sub("^mscale/output/decode/conv", "module.ocr.cls_head", name)
        name = re.sub("^mscale/attention/([0-9]*)/([a-z]*)", lambda m: f"module.scale_attn.{m.group(2)}{int(m.group(1)) - 1}", name)
        name = re.sub("^mscale/attention/decode/conv", "module.scale_attn.conv2", name)

    return name

def create(input=None, sequential=True):
    return_model = input is None
    if input is None:
        input = tf.keras.layers.Input((None, None, 3))

    def block(x, *args, name=None, config=tfcv.model.config.Config(), **kwargs):
        x = conv_norm_act(x, *args, name=name, config=config, **kwargs)
        x = (psa.sequential if sequential else psa.parallel)(x, reduction=4, name=join(name, "psa"), fix_bias=False, config=config)
        return x
    basic_block = partial(resnet.basic_block_v1, block=block)

    x = input
    x = hrnet.hrnet(x,
        num_units=[[4], [4, 4], [4, 4, 4], [4, 4, 4, 4]],
        filters=[[64], [48, 96], [48, 96, 192], [48, 96, 192, 384]],
        blocks=[resnet.bottleneck_block_v1, basic_block, basic_block, basic_block],
        num_modules=[1, 1, 4, 3],
        stem=True,
        config=config,
    )
    x = ocr.ocr(x, regions=19, filters=512, filters_qkv=256, fix_bias_before_norm=False, config=config)
    x = conv_norm_act(x, filters=512, kernel_size=1, stride=1, bias=False, name=join("final"), config=config)
    output, weights = mscale.mscale_decode(x, filters=19, filters_mid=256, shape=tf.shape(input)[1:-1], dropout=0.05, name="mscale", config=config)

    model = tf.keras.Model(inputs=[input], outputs=[output, weights])

    if sequential:
        url = "https://cmu.app.box.com/s/if90kw6r66q2y6c5xparflhnbwi6c2yi"
        name = "best_checkpoint_86.76_PSA_s.pth"
    else:
        url = "https://cmu.app.box.com/s/uyzzfmkx8p2ipcznpzdtf14ng63s65sq"
        name = "best_checkpoint_86.98_PSA_p.pth"
    weights = download_cmu_box(url, name)
    tfcv.model.pretrained.weights.load_pth(weights, model, convert_name)

    return model if return_model else x
