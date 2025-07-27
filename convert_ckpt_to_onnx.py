import os
import torch
import pytorch_lightning as pl
from model_new import IJEPA  # Ensure this import matches your project structure

def convert_ckpt_to_onnx(checkpoint_dir):
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith('.ckpt'):
                ckpt_path = os.path.join(root, file)
                try:
                    checkpoint = torch.load(ckpt_path, weights_only=True)
                    model = IJEPA.load_from_checkpoint(ckpt_path)
                    model.eval()
                    
                    # Get necessary parameters for dummy input
                    num_patches = (model.hparams.img_size // model.hparams.patch_size) ** 2
                    embed_dim = model.hparams.embed_dim
                    
                    # Create dummy input for ONNX export: [batch_size, num_patches, embed_dim]
                    dummy_input = torch.randn(1, num_patches, embed_dim).to(next(model.parameters()).device)
                    
                    # Export student encoder
                    student_onnx_path = os.path.join(root, file.replace('.ckpt', '-student.onnx'))
                    torch.onnx.export(
                        model.model.student_encoder,
                        dummy_input,
                        student_onnx_path,
                        input_names=["input"],
                        output_names=["output"],
                        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                    )
                    print(f"Exported student ONNX to {student_onnx_path}")
                    
                    # Export teacher encoder
                    teacher_onnx_path = os.path.join(root, file.replace('.ckpt', '-teacher.onnx'))
                    torch.onnx.export(
                        model.model.teacher_encoder,
                        dummy_input,
                        teacher_onnx_path,
                        input_names=["input"],
                        output_names=["output"],
                        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                    )
                    print(f"Exported teacher ONNX to {teacher_onnx_path}")
                except Exception as e:
                    print(f"Error processing {ckpt_path}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert CKPT files to ONNX")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory containing CKPT files")
    args = parser.parse_args()
    convert_ckpt_to_onnx(args.checkpoint_dir) 