import os
import torch
torch.manual_seed(1234)
import random
random.seed(1234)

import numpy as np
from pathlib import Path
import soundfile as sf
import torchaudio
from loguru import logger
from tools.llama.generate import (
    load_model as load_llama_model,
    generate_long,
    encode_tokens,
    BaseTransformer
)
from tools.vqgan.inference import load_model as load_vqgan_model

class FishSpeechInference:
    def __init__(self, checkpoint_dir="checkpoints/fish-speech-1.5", device='cuda'):
        self.base_checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir = self.base_checkpoint_dir
            
        self.project_root = Path(__file__).parent.absolute()
        self.vqgan_model = self._init_vqgan_model(device)
        self.llama_model, self.decode_one_token = self._init_llama_model(device)
    
    def _init_vqgan_model(self, device):
        model = load_vqgan_model(
            config_name="firefly_gan_vq",
            checkpoint_path=self.get_vqgan_model_path(),
            device=device
        )
        return model
    
    def _init_llama_model(self, device):
        model, decode_one_token = load_llama_model(
            checkpoint_path=str(self.checkpoint_dir),
            device=device,
            precision=torch.bfloat16,
            compile=False
        )
        
        # 캐시 설정
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype
            )
        return model, decode_one_token
    
    def get_vqgan_model_path(self):
        """VQGAN 모델 경로 반환"""
        return str(self.base_checkpoint_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
    
    def generate_voice_prompt(self, input_audio):
        """입력 오디오로부터 음성 프롬프트 생성"""
        # 오디오 로드 및 전처리
        audio, sr = torchaudio.load(str(input_audio))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(
            audio, 
            sr, 
            self.vqgan_model.spec_transform.sample_rate
        )
        
        # VQGAN으로 처리
        audios = audio[None].to("cuda")
        audio_lengths = torch.tensor([audios.shape[2]], device="cuda", dtype=torch.long)
        indices = self.vqgan_model.encode(audios, audio_lengths)[0][0]
        
        # 결과 저장
        output_path = Path("fake.npy")
        np.save(output_path, indices.cpu().numpy())
        return output_path
    
    def generate_semantic_tokens(self, text, prompt_text, prompt_tokens):
        """LLaMA 모델을 사용하여 시맨틱 토큰 생성"""
        # numpy array를 torch tensor로 변환
        prompt_tokens_data = np.load(prompt_tokens)
        prompt_tokens_tensor = torch.from_numpy(prompt_tokens_data)
        
        # 입력 토큰화
        encoded = encode_tokens(
            tokenizer=self.llama_model.tokenizer,
            string=text,
            device="cuda",
            prompt_tokens=prompt_tokens_tensor,
            num_codebooks=self.llama_model.config.num_codebooks
        )
        
        # 토큰 생성
        responses = []
        generator = generate_long(
            model=self.llama_model,
            device="cuda",
            decode_one_token=self.decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=0,
            temperature=0.3,
            # compile=True,
            top_p=1,
            repetition_penalty=1.13,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens_tensor,
            iterative_prompt=False  # 텍스트를 나누지 않고 한번에 처리
        )
        
        idx = 0
        codes = []
        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)
            elif response.action == "next":
                if codes:
                    output_path = Path(f"codes_{idx}.npy")
                    np.save(output_path, torch.cat(codes, dim=1).cpu().numpy())
                    responses.append(output_path)
                codes = []
                idx += 1
                
        return responses
    
    def generate_audio(self, semantic_tokens):
        """시맨틱 토큰으로부터 오디오 생성"""
        # 토큰 로드
        indices = np.load(semantic_tokens)
        indices = torch.from_numpy(indices).to("cuda").long()
        
        # 오디오 생성
        feature_lengths = torch.tensor([indices.shape[1]], device="cuda")
        fake_audios, _ = self.vqgan_model.decode(
            indices=indices[None],
            feature_lengths=feature_lengths
        )
        
        # 결과 저장
        output_path = Path(semantic_tokens).with_suffix('.wav')
        fake_audio = fake_audios[0, 0].detach().float().cpu().numpy()
        sf.write(
            output_path,
            fake_audio,
            self.vqgan_model.spec_transform.sample_rate
        )
        return output_path
    
    def run_inference(self, text, reference_audio, prompt_text):
        """전체 추론 파이프라인 실행"""
        os.chdir(self.project_root)
        
        # 1. 음성 프롬프트 생성
        prompt_tokens = self.generate_voice_prompt(reference_audio)
        logger.info(f"Generated voice prompt: {prompt_tokens}")
        
        # 2. 시맨틱 토큰 생성
        code_files = self.generate_semantic_tokens(text, prompt_text, prompt_tokens)
        logger.info(f"Generated semantic tokens: {code_files}")
        
        # 3. 오디오 생성
        outputs = []
        for code_file in code_files:
            output_path = self.generate_audio(code_file)
            outputs.append(output_path)
            logger.info(f"Generated audio: {output_path}")
            
        return outputs

def main():
    # 테스트용 입력값
    text = "이게정말로가능하다고하면.내가너앞에서겠다.아니무슨말을하면알아들어야지.이게뭐하는짓이야"
    reference_audio = r"C:\Users\kdrkdrkdr\Desktop\lightweight-fish-speech-zeroshot\park.wav"
    prompt_text = "간단하게 말씀 나누시죠. 이 모든 내용이, 사실입니까. 먼저 신랑에게 묻겠습니다. 저는, 지금부터 제가 할 수 있는 일을 하러 갈 겁니다. 보는 눈이 있으니, 일단 자리를 이동하시죠. 매니저들의 인맥과 노하우를 활용해서 성사시키기 어려운 계약을 따내거나, 부득이하게 겹친 스케줄을 풀기도 하죠. 꼭 매니저가 해야 한다는 법은 없습니다. 지금처럼 회사 차원에서 관리하기도 합니다. 저녁 축하드립니다. 만약 계속 일을 한다면, 세 가지의 결말이 있습니다. "
    checkpoint_dir = "checkpoints/fish-speech-1.5"
    
    inferencer = FishSpeechInference(checkpoint_dir=checkpoint_dir, device='cuda')
    output_files = inferencer.run_inference(text, reference_audio, prompt_text)
    logger.info(f"최종 생성된 오디오 파일들: {output_files}")

if __name__ == "__main__":
    main()
