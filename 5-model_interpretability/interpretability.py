import torch
import numpy as np
import re
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

############################################################
# 1. Load Model
############################################################

model_path = "Your trained model path"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",#"auto""cuda:0"
    attn_implementation="eager"
)
model.eval()

############################################################
# 2. Prompt
############################################################

instruction = "Predict the compressive yield strength at room temperature for the following refractory high entropy alloy."
# instruction = "Predict the density for the following refractory high entropy alloy."

input_text = "Al14.6Nb26.9V33.5Zr8.1Mo10.9 alloy prepared by vacuum arc melting with ingots re-melted five times."
# input_text = "Mo22.22Nb22.22V22.22Ta11.11Cr22.22 alloy prepared by mechanical alloying for 60h at 350rpm under Ar gas protection, followed by spark plasma sintering at 1700°C and axial pressure of 40MPa for 10min."
prompt = (
    "<|im_start|>system\nYou are a materials science expert.<|im_end|>\n"
    f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
prompt_len = inputs.input_ids.shape[1]

############################################################
# 3. Generate WITH attention tracing
############################################################

gen_outputs = model.generate(
    **inputs,
    generation_config=GenerationConfig(
        max_new_tokens=64,
        repetition_penalty=1, 
        temperature=0.6, 
        top_p=0.9, 
        top_k=50,
        do_sample=False,
        output_attentions=True,
        return_dict_in_generate=True
    )
)

generated_ids = gen_outputs.sequences[0]

decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
print("\nMODEL OUTPUT:\n", decoded)

############################################################
# 4. Collect ALL generation attentions
############################################################
gen_attentions = gen_outputs.attentions

num_layers = len(gen_attentions[0])
num_heads = gen_attentions[0][0].shape[1]

all_step_attn = []

for step in gen_attentions:

    layer_stack = []

    for l in range(num_layers):

        attn = step[l][0, :, 0, :prompt_len]  # (head, prompt_len)
        layer_stack.append(attn)

    layer_stack = torch.stack(layer_stack)  # (layer, head, prompt_len)
    all_step_attn.append(layer_stack)

# average over ALL generated tokens
full_output_attn = torch.stack(all_step_attn).sum(0)
prompt_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

print("Attention shape:", full_output_attn.shape)

output_dir = "raw_attention_data"
os.makedirs(output_dir, exist_ok=True)
# Save full attention tensor (layer, head, prompt_len)
torch.save(full_output_attn.cpu(), os.path.join(output_dir, "full_output_attn.pt"))
metadata = {
    "model_path": model_path,
    "instruction": instruction,
    "input_text": input_text,
    "prompt": prompt,
    "prompt_len": prompt_len,
    "prompt_tokens": prompt_tokens,
    "num_layers": num_layers,
    "num_heads": num_heads,
    "generated_output": decoded.strip(),
}
with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)


############################################################
# 5. TOKEN PROCESSING (YOUR LOGIC — FIXED)
############################################################

def split_newline_tokens(tokens):

    new_tokens = []

    for tok in tokens:
        if "Ċ" in tok:
            parts = tok.split("Ċ")

            for i, p in enumerate(parts):
                if p:
                    new_tokens.append(p)
                if i < len(parts) - 1:
                    new_tokens.append("Ċ")  # newline boundary
        else:
            new_tokens.append(tok)

    return new_tokens


def clean_token(tok):

    tok = tok.replace("<|im_start|>", "")
    tok = tok.replace("<|im_end|>", "")

    tok = tok.replace("▁", " ")
    tok = tok.replace("Ġ", " ")

    tok = tok.replace("Â", "")  # fix ° encoding

    tok = re.sub(r"\s+", " ", tok)

    return tok.strip()


############################################################
# merge_tokens (newline SAFE)
############################################################

ELEMENT_PATTERN = re.compile(r'^[A-Z][a-z]?$')
NUMBER_PATTERN = re.compile(r'^[0-9.]$')


def merge_tokens(tokens):

    tokens = split_newline_tokens(tokens)

    words = []
    mapping = []

    current_word = ""
    current_indices = []

    # ===== alloy builder =====
    alloy_word = ""
    alloy_indices = []
    building_alloy = False

    def flush_normal():
        nonlocal current_word, current_indices
        if current_word:
            words.append(current_word)
            mapping.append(current_indices)
            current_word = ""
            current_indices = []

    def flush_alloy():
        nonlocal alloy_word, alloy_indices, building_alloy
        if alloy_word:
            words.append(alloy_word)
            mapping.append(alloy_indices)
        alloy_word = ""
        alloy_indices = []
        building_alloy = False

    for i, tok in enumerate(tokens):

        # newline boundary
        if tok == "Ċ":
            flush_normal()
            flush_alloy()
            continue

        clean_tok = clean_token(tok)

        if clean_tok == "":
            continue

        # ==================================================
        # ⭐ ALLOY DETECTION
        # ==================================================

        # element symbol
        if ELEMENT_PATTERN.match(clean_tok):

            flush_normal()
            flush_alloy()

            alloy_word = clean_tok
            alloy_indices = [i]
            building_alloy = True
            continue

        # number continuation
        if building_alloy and NUMBER_PATTERN.match(clean_tok):

            alloy_word += clean_tok
            alloy_indices.append(i)
            continue

        # alloy finished
        if building_alloy:
            flush_alloy()

        # ==================================================
        # NORMAL WORD MERGING
        # ==================================================

        if tok.startswith(("Ġ", "▁")):

            flush_normal()

            current_word = clean_tok
            current_indices = [i]

        else:
            current_word += clean_tok
            current_indices.append(i)

    flush_alloy()
    flush_normal()

    return words, mapping


############################################################
# Remove role words
############################################################

def filter_words(words, mapping):

    ignore = {"system", "user", "assistant"}

    new_words = []
    new_mapping = []

    for w, m in zip(words, mapping):
        if w.lower() not in ignore:
            new_words.append(w)
            new_mapping.append(m)

    return new_words, new_mapping


############################################################
# Apply merging
############################################################

prompt_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

words, word_token_map = merge_tokens(prompt_tokens)
words, word_token_map = filter_words(words, word_token_map)

print("\nWORDS:")
print(words)

############################################################
# 6. TOKEN → WORD ATTENTION
############################################################

def token_to_word_attention(attn, mapping):

    word_attn = []
    seq_len = attn.shape[-1]

    for token_ids in mapping:

        valid_ids = [i for i in token_ids if i < seq_len]

        if len(valid_ids) == 0:
            word_attn.append(torch.zeros(attn.shape[0], device=attn.device))
        else:
            word_attn.append(attn[:, valid_ids].mean(-1))######

    return torch.stack(word_attn, dim=-1)


############################################################
# Build word attention tensor
############################################################

word_attentions = []

for l in range(num_layers):

    layer_heads = []

    for h in range(num_heads):

        token_attn = full_output_attn[l, h]

        word_attn = token_to_word_attention(
            token_attn.unsqueeze(0),
            word_token_map
        )[0]

        layer_heads.append(word_attn)

    word_attentions.append(torch.stack(layer_heads))

word_attentions = torch.stack(word_attentions)

print("Word attention shape:", word_attentions.shape)
# (layer, head, words)

import os
from matplotlib.colors import Normalize
import numpy as np

############################################################
# COLOR UTIL
############################################################

# COLOR_LOW = np.array([134, 186, 227])   # #86BAE3
COLOR_LOW = np.array([255, 255, 255])   # #86BAE3
COLOR_HIGH = np.array([251, 126, 112])  # #FB7E70


def color_from_score(score, vmin, vmax):
    """
    map attention → white → #FB7E70 gradient
    """
    if vmax - vmin < 1e-8:
        alpha = 0
    else:
        alpha = (score - vmin) / (vmax - vmin)

    # Clamp alpha to [0, 1] for safety
    alpha = max(0.0, min(1.0, alpha))
    
    rgb = (1 - alpha) * COLOR_LOW + alpha * COLOR_HIGH
    r, g, b = rgb.astype(int)
    return f"rgb({r},{g},{b})"


############################################################
# MAIN HTML VISUALIZATION
############################################################

def save_attention_html(
    words,
    word_attentions,
    layer=20,
    head=5,
    save_path="attention"
):
    """
    Save colored attention sentence as HTML with responsive layout and colorbar
    positioned at the bottom-right of the text block.
    """

    attn = word_attentions[layer, head].detach().cpu().numpy()

    vmin = float(attn.min())
    vmax = float(attn.max())

    # --------------------------------------------------
    # Build colored text
    # --------------------------------------------------
    html_words = ""
    for w, s in zip(words[6:], attn[6:]):
        bg_color = color_from_score(s, vmin, vmax)
        html_words += (
            f'<span class="token" style="background-color:{bg_color};">{w}</span> '
        )

    # --------------------------------------------------
    # COLORBAR (CSS gradient)
    # --------------------------------------------------
    colorbar_html = f"""
    <div class="colorbar-wrapper">
        <div class="colorbar"></div>
        <div class="colorbar-labels">
            <span>{vmin:.2f}</span>
            <span>attention</span>
            <span>{vmax:.2f}</span>
        </div>
    </div>
    """

    # --------------------------------------------------
    # FULL HTML
    # --------------------------------------------------
    html = f"""
    <html>
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background: white;
        }}

        .container {{
            display: inline-block;
            max-width: 100%;
            position: relative;
        }}

        .sentence {{
            font-size: 28px;
            line-height: 1.5;
            word-break: break-word;
            overflow-wrap: break-word;
            white-space: normal;
        }}

        .token {{
            color: black;
            padding: 2px 2px;
            margin: 1px;
            border-radius: 8px;
            display: inline-block;
            white-space: nowrap;
        }}

        .colorbar-wrapper {{
            margin-top: 12px;
            width: 260px;
            float: right;
        }}

        .colorbar {{
            height: 20px;
            border-radius: 8px;
            background: linear-gradient(
                to right,
                rgb({COLOR_LOW[0]},{COLOR_LOW[1]},{COLOR_LOW[2]}),
                rgb({COLOR_HIGH[0]},{COLOR_HIGH[1]},{COLOR_HIGH[2]})
            );
            border: 1px solid #ccc;
        }}

        .colorbar-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 22px;
            margin-top: 4px;
        }}

        /* Clear float */
        .container::after {{
            content: "";
            display: table;
            clear: both;
        }}
    </style>
    </head>
    <body>
        <div class="container">
            <div class="sentence">
                {html_words}
            </div>
            {colorbar_html}
        </div>
    </body>
    </html>
    """

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path,f"attention-layer{layer}-head{head}.html")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ HTML saved to: {os.path.abspath(save_path)}")

save_attention_html(
    words,
    word_attentions,
    layer=26,
    head=1,
    save_path=output_dir
)

# Compute layer-wise total attention (sum over heads)
layer_total_attn = word_attentions.mean(dim=1)  # Shape: (num_layers, num_words)

# Convert to numpy for plotting
layer_total_attn_np = layer_total_attn.detach().cpu().numpy()
# Save the layer attention to ecxel
np.savetxt(os.path.join(output_dir, "layer_attention.csv"), layer_total_attn_np, delimiter=",")

import matplotlib.pyplot as plt
import seaborn as sns

def plot_layer_attention_heatmap(layer_total_attn, words, save_path=None):
    """Plot layer-wise attention as heatmap"""
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        layer_total_attn.T[:,10:],
        xticklabels=[f"Layer {i}" for i in range(10,len(layer_total_attn))],
        yticklabels=words,
        cmap="viridis",
        cbar_kws={'label': 'Total Attention Score'}
    )
    plt.title("Layer-wise Total Attention (Mean over Heads)")
    plt.xlabel("Layers")
    plt.ylabel("Words")
    plt.xticks(rotation=90, ha='center')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()



# Plot full heatmap
plot_layer_attention_heatmap(
    layer_total_attn_np, 
    words, 
    save_path=os.path.join(output_dir, "layer_attention_heatmap.png")
)


