import torch
import os

# Append all files from /home1/Saurabh/tools/espnet
import sys
sys.path.append("/home1/Saurabh/tools/espnet")

# Import the model
from espnet2.asr.encoder.attention_encoder_old import CombinedEncoder

# Define a dummy input and lengths for testing
def get_dummy_inputs(batch_size, seq_len, feature_dim, device):
    enc_out1 = torch.randn(batch_size, seq_len, 256, device=device)
    enc_out2 = torch.randn(batch_size, seq_len, 64, device=device)
    encoder_out_lens = torch.randint(1, seq_len + 1, (batch_size,), device=device)
    return enc_out1, enc_out2, encoder_out_lens

# Initialize the model
output_dim = 256
feature_dim = 256 + 64
num_heads = 8
hidden_dim = 512
num_layers = 2
num_dialects = 4
dropout_rate = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CombinedEncoder(
    output_dim=output_dim,
    feature_dim=feature_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_dialects=num_dialects,
    dropout_rate=dropout_rate
).to(device)

# Generate dummy inputs
batch_size = 4
seq_len = 50
enc_out1, enc_out2, encoder_out_lens = get_dummy_inputs(batch_size, seq_len, feature_dim, device)

# Export model to ONNX
onnx_path = "attention_encoder_old.onnx"
torch.onnx.export(
    model,
    (enc_out1, enc_out2, encoder_out_lens),
    onnx_path,
    opset_version=13,  # Updated to ensure support for aten::unflatten
    input_names=["enc_out1", "enc_out2", "encoder_out_lens"],
    output_names=["logits"],
    dynamic_axes={"enc_out1": {0: "batch_size", 1: "seq_len"},
                  "enc_out2": {0: "batch_size", 1: "seq_len"},
                  "encoder_out_lens": {0: "batch_size"},
                  "logits": {0: "batch_size"}}
)

print(f"Model exported to ONNX format at '{onnx_path}'")

# Visualize with Netron
try:
    import netron
    netron.start(onnx_path)
except ImportError:
    print("Netron is not installed. Please install it using 'pip install netron' and run this script again.")
