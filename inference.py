import os
import argparse
import subprocess
from pathlib import Path

class FishSpeechInference:
    def __init__(self, checkpoint_dir="checkpoints/fish-speech-1.5", project_name=None):
        # Base directory for VQGAN model
        self.base_checkpoint_dir = Path(checkpoint_dir)
        
        # Project-specific directory for other models
        if project_name:
            self.checkpoint_dir = Path(checkpoint_dir.replace("fish-speech-1.5", f"fish-speech-1.5-{project_name}-lora"))
        else:
            self.checkpoint_dir = self.base_checkpoint_dir
            
        self.project_root = Path(__file__).parent.absolute()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def check_models(self):
        """Check if required model files exist"""
        vqgan_path = self.base_checkpoint_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        if not vqgan_path.exists():
            raise FileNotFoundError(
                f"Required VQGAN model file not found:\n"
                f"{vqgan_path}\n"
                f"Please ensure the file is present in the base checkpoint directory."
            )
    
    def get_vqgan_model_path(self):
        """Get VQGAN model path from base directory"""
        return str(self.base_checkpoint_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
    
    def generate_voice_prompt(self, input_audio):
        """Generate voice prompt from reference audio"""
        print(f"Generating voice prompt from {input_audio}...")
        subprocess.run([
            "python", "tools/vqgan/inference.py",
            "-i", str(input_audio),
            "--checkpoint-path", self.get_vqgan_model_path()
        ], check=True)
        return Path("fake.npy")
    
    def generate_semantic_tokens(self, text, prompt_text=None, prompt_tokens=None, 
                               num_samples=1, use_compile=False, use_half=False):
        """Generate semantic tokens from text"""
        print(f"Generating semantic tokens for text: {text}")
        
        cmd = [
            "python", "tools/llama/generate.py",
            "--text", text,
            "--checkpoint-path", str(self.checkpoint_dir),
            "--num-samples", str(num_samples)
        ]
        
        if prompt_text:
            cmd.extend(["--prompt-text", prompt_text])
        if prompt_tokens:
            cmd.extend(["--prompt-tokens", str(prompt_tokens)])
        if use_compile:
            cmd.append("--compile")
        if use_half:
            cmd.append("--half")
            
        subprocess.run(cmd, check=True)
        return list(Path(".").glob("codes_*.npy"))
    
    def generate_audio(self, semantic_tokens):
        """Generate audio from semantic tokens"""
        print(f"Generating audio from semantic tokens: {semantic_tokens}")
        subprocess.run([
            "python", "tools/vqgan/inference.py",
            "-i", str(semantic_tokens),
            "--checkpoint-path", self.get_vqgan_model_path()
        ], check=True)
    
    def run_inference(self, text, reference_audio=None, prompt_text=None, 
                     num_samples=1, use_compile=False, use_half=False):
        """Run complete inference pipeline"""
        try:
            # Validate prompt inputs
            if reference_audio and not prompt_text:
                raise ValueError("When using reference audio, you must also provide the prompt text via --prompt-text")
            
            # Check if required models exist
            self.check_models()
            
            # Change to project root directory
            original_dir = os.getcwd()
            os.chdir(self.project_root)
            
            try:
                # Generate voice prompt if reference audio is provided
                prompt_tokens = None
                if reference_audio:
                    prompt_tokens = self.generate_voice_prompt(reference_audio)
                
                # Generate semantic tokens
                code_files = self.generate_semantic_tokens(
                    text=text,
                    prompt_text=prompt_text,
                    prompt_tokens=prompt_tokens,
                    num_samples=num_samples,
                    use_compile=use_compile,
                    use_half=use_half
                )
                
                # Generate audio for each semantic token file
                for code_file in code_files:
                    self.generate_audio(code_file)
                    
                print("Inference completed successfully!")
                print(f"Generated {len(code_files)} audio samples")
                
            finally:
                # Change back to original directory
                os.chdir(original_dir)
                
        except subprocess.CalledProcessError as e:
            print(f"Error occurred during execution: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Fish-Speech Inference")
    parser.add_argument("--text", type=str, required=True,
                      help="Text to convert to speech")
    parser.add_argument("--reference-audio", type=str,
                      help="Reference audio file for voice prompt (optional)")
    parser.add_argument("--prompt-text", type=str,
                      help="Reference text for prompt (optional)")
    parser.add_argument("--checkpoint-dir", type=str,
                      default="checkpoints/fish-speech-1.5",
                      help="Directory for model checkpoints")
    parser.add_argument("--project-name", type=str,
                      help="Project name for fine-tuned model (e.g., 'test' for fish-speech-1.5-test-lora)")
    parser.add_argument("--num-samples", type=int, default=1,
                      help="Number of samples to generate")
    parser.add_argument("--compile", action="store_true",
                      help="Enable CUDA kernel fusion (requires C compiler)")
    parser.add_argument("--half", action="store_true",
                      help="Use half precision (for GPUs without bf16 support)")
    
    args = parser.parse_args()
    
    inferencer = FishSpeechInference(
        checkpoint_dir=args.checkpoint_dir,
        project_name=args.project_name
    )
    
    inferencer.run_inference(
        text=args.text,
        reference_audio=args.reference_audio,
        prompt_text=args.prompt_text,
        num_samples=args.num_samples,
        use_compile=args.compile,
        use_half=args.half
    )

if __name__ == "__main__":
    main()
