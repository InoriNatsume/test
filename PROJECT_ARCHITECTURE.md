# NewBie Image Exp0.1 - 프로젝트 아키텍처 문서

## 프로젝트 개요

**NewBie image Exp0.1**은 Next-DiT에 기반한 3.5B 파라미터 규모의 DiT(Diffusion Transformer) 모델로, 텍스트-이미지 생성을 위한 실험적 프레임워크입니다.

---

## 1. 폴더 구조

```mermaid
graph TD
    Root[NewBie-image-Exp0.1] --> configs[configs/]
    Root --> data[data/]
    Root --> models[models/]
    Root --> transport[transport/]
    Root --> scripts[scripts/]
    Root --> util[util/]
    
    Root --> finetune[finetune.py<br/>파인튜닝 메인 스크립트]
    Root --> sample[sample.py<br/>샘플링 스크립트]
    Root --> demo[demo.py<br/>Gradio 데모]
    Root --> parallel[parallel.py<br/>분산 학습]
    Root --> imgproc[imgproc.py<br/>이미지 전처리]
    Root --> grad_norm[grad_norm.py<br/>그래디언트 정규화]
    
    configs --> data_yaml[data.yaml]
    
    data --> data_init[__init__.py]
    data --> dataset[dataset.py<br/>데이터셋 클래스]
    data --> data_reader[data_reader.py<br/>데이터 읽기]
    
    models --> models_init[__init__.py]
    models --> model[model.py<br/>메인 모델]
    models --> components[components.py<br/>RMSNorm 등]
    
    transport --> transport_init[__init__.py]
    transport --> transport_py[transport.py<br/>Transport 클래스]
    transport --> dpm_solver[dpm_solver.py<br/>DPM Solver]
    transport --> integrators[integrators.py<br/>ODE/SDE 적분기]
    transport --> path[path.py<br/>Path 정의]
    transport --> utils[utils.py<br/>유틸리티]
    
    scripts --> run_finetune[run_1024_finetune.sh]
    scripts --> sample_sh[sample.sh]
    
    util --> misc[misc.py]
```

---

## 2. 주요 클래스 다이어그램

### 2.1 모델 구조 (models/)

```mermaid
classDiagram
    class NextDiT {
        +int in_channels
        +int out_channels
        +int hidden_size
        +int patch_size
        +List~int~ num_layers
        +__init__()
        +forward(x, t, y, clip_emb, mask)
        +forward_with_cfg()
        +initialize_weights()
    }
    
    class TimestepEmbedder {
        +int hidden_size
        +int frequency_embedding_size
        +__init__()
        +timestep_embedding(t, dim)
        +forward(t)
    }
    
    class JointAttention {
        +int dim
        +int n_heads
        +int n_kv_heads
        +bool qk_norm
        +__init__()
        +forward(x, x_mask, freqs_cis)
        +apply_rotary_emb(x_in, freqs_cis)
        +_upad_input()
    }
    
    class FeedForward {
        +int dim
        +int hidden_dim
        +__init__()
        +forward(x)
        +_forward_silu_gating(x1, x3)
    }
    
    class TransformerBlock {
        +JointAttention attn
        +FeedForward mlp
        +RMSNorm norm1
        +RMSNorm norm2
        +__init__()
        +forward(x, x_mask, freqs_cis)
    }
    
    class RMSNorm {
        +float eps
        +Parameter weight
        +__init__(dim, eps)
        +_norm(x)
        +forward(x)
    }
    
    class PatchEmbed {
        +int patch_size
        +int in_chans
        +int embed_dim
        +__init__()
        +forward(x)
    }
    
    NextDiT --> TimestepEmbedder : uses
    NextDiT --> PatchEmbed : uses
    NextDiT --> TransformerBlock : contains
    TransformerBlock --> JointAttention : contains
    TransformerBlock --> FeedForward : contains
    TransformerBlock --> RMSNorm : uses
```

### 2.2 데이터 처리 (data/)

```mermaid
classDiagram
    class MyDataset {
        +List annotations
        +ItemProcessor item_processor
        +bool cache_on_disk
        +__init__(config_path, item_processor)
        +__len__()
        +__getitem__(index)
        +_collect_annotations()
        +assign_buckets(crop_size_list)
        +groups()
    }
    
    class ItemProcessor {
        <<abstract>>
        +process_item(data_item, training_mode)
    }
    
    class T2IItemProcessor {
        +transform image_transform
        +set special_format_set
        +__init__(transform)
        +process_item(data_item, training_mode)
    }
    
    class DataBriefReportException {
        +str message
        +__init__(message)
    }
    
    class DataNoReportException {
        +str message
        +__init__(message)
    }
    
    MyDataset --> ItemProcessor : uses
    T2IItemProcessor ..|> ItemProcessor : implements
    MyDataset ..> DataBriefReportException : throws
    MyDataset ..> DataNoReportException : throws
```

### 2.3 Transport 시스템 (transport/)

```mermaid
classDiagram
    class Transport {
        +ModelType model_type
        +Path path
        +WeightType loss_type
        +__init__()
        +sample(x1)
        +training_losses(model, x1)
        +get_drift()
        +get_score()
        +prior_logp(z)
    }
    
    class Sampler {
        +Transport transport
        +__init__(transport)
        +sample_ode()
        +sample_sde()
        +sample_dpm()
        +sample_ode_likelihood()
    }
    
    class ModelType {
        <<enumeration>>
        NOISE
        SCORE
        VELOCITY
    }
    
    class PathType {
        <<enumeration>>
        LINEAR
        GVP
        VP
    }
    
    class WeightType {
        <<enumeration>>
        NONE
        VELOCITY
        LIKELIHOOD
    }
    
    class DPM_Solver {
        +NoiseScheduleFlow noise_schedule
        +__init__()
        +sample()
    }
    
    Transport --> ModelType : uses
    Transport --> PathType : uses
    Transport --> WeightType : uses
    Sampler --> Transport : uses
    Sampler --> DPM_Solver : uses
```

---

## 3. 실행 흐름도

### 3.1 학습 프로세스 (finetune.py)

```mermaid
flowchart TD
    Start([시작]) --> ParseArgs[명령행 인자 파싱]
    ParseArgs --> InitDist[분산 학습 초기화<br/>distributed_init]
    InitDist --> LoadModel[모델 로드<br/>NextDiT]
    LoadModel --> LoadTextEncoder[텍스트 인코더 로드<br/>Gemma3-4B-it + Jina CLIP v2]
    LoadTextEncoder --> LoadVAE[VAE 로드<br/>FLUX.1-dev 16ch]
    LoadVAE --> SetupFSDP[FSDP 설정<br/>setup_fsdp_sync]
    SetupFSDP --> CreateDataset[데이터셋 생성<br/>MyDataset]
    CreateDataset --> CreateDataloader[데이터로더 생성<br/>with bucket sampler]
    CreateDataloader --> CreateOptimizer[옵티마이저 생성<br/>AdamW]
    CreateOptimizer --> TrainLoop{학습 루프}
    
    TrainLoop --> GetBatch[배치 가져오기]
    GetBatch --> EncodePrompt[프롬프트 인코딩<br/>encode_prompt]
    EncodePrompt --> EncodeImage[이미지 인코딩<br/>VAE.encode]
    EncodeImage --> SampleTimestep[타임스텝 샘플링]
    SampleTimestep --> ForwardModel[모델 순전파<br/>model.forward]
    ForwardModel --> ComputeLoss[손실 계산<br/>transport.training_losses]
    ComputeLoss --> Backward[역전파]
    Backward --> UpdateWeights[가중치 업데이트]
    UpdateWeights --> UpdateEMA[EMA 업데이트<br/>update_ema]
    UpdateEMA --> SaveCheckpoint{체크포인트 저장?}
    SaveCheckpoint -->|Yes| Save[체크포인트 저장]
    SaveCheckpoint -->|No| CheckContinue{계속?}
    Save --> CheckContinue
    CheckContinue -->|Yes| TrainLoop
    CheckContinue -->|No| End([종료])
```

### 3.2 샘플링 프로세스 (sample.py)

```mermaid
flowchart TD
    Start([시작]) --> ParseArgs[명령행 인자 파싱]
    ParseArgs --> InitDist[분산 초기화]
    InitDist --> LoadModel[모델 로드<br/>NextDiT]
    LoadModel --> LoadCheckpoint[체크포인트 로드]
    LoadCheckpoint --> LoadTextEncoder[텍스트 인코더 로드]
    LoadTextEncoder --> LoadVAE[VAE 로드]
    LoadVAE --> CreateTransport[Transport 생성<br/>create_transport]
    CreateTransport --> CreateSampler[Sampler 생성]
    CreateSampler --> SampleLoop{샘플링 루프}
    
    SampleLoop --> GetPrompt[프롬프트 가져오기]
    GetPrompt --> EncodePrompt[프롬프트 인코딩]
    EncodePrompt --> InitNoise[노이즈 초기화<br/>randn]
    InitNoise --> ODESampling[ODE 샘플링<br/>sampler.sample_ode]
    ODESampling --> DecodeLatent[잠재변수 디코딩<br/>VAE.decode]
    DecodeLatent --> SaveImage[이미지 저장]
    SaveImage --> CheckContinue{계속?}
    CheckContinue -->|Yes| SampleLoop
    CheckContinue -->|No| End([종료])
```

### 3.3 데모 프로세스 (demo.py)

```mermaid
flowchart TD
    Start([시작]) --> ParseArgs[명령행 인자 파싱]
    ParseArgs --> StartModelProcess[모델 프로세스 시작<br/>multiprocessing]
    StartModelProcess --> LoadGradio[Gradio UI 로드]
    LoadGradio --> WaitUser{사용자 입력 대기}
    
    WaitUser --> GetInput[입력 받기<br/>prompt, params]
    GetInput --> QueueRequest[요청 큐에 삽입]
    QueueRequest --> ModelProcess[모델 프로세스 처리]
    
    ModelProcess --> EncodePrompt2[프롬프트 인코딩]
    EncodePrompt2 --> GenerateImage[이미지 생성]
    GenerateImage --> ReturnResult[결과 반환]
    ReturnResult --> DisplayUI[UI에 표시]
    DisplayUI --> WaitUser
```

---

## 4. 컴포넌트 간 의존성

```mermaid
graph LR
    subgraph "Main Scripts"
        A[finetune.py]
        B[sample.py]
        C[demo.py]
    end
    
    subgraph "Core Models"
        D[models/model.py<br/>NextDiT]
        E[models/components.py<br/>RMSNorm]
    end
    
    subgraph "Data Pipeline"
        F[data/dataset.py<br/>MyDataset]
        G[data/data_reader.py<br/>read_general]
    end
    
    subgraph "Transport System"
        H[transport/transport.py<br/>Transport]
        I[transport/dpm_solver.py<br/>DPM_Solver]
        J[transport/integrators.py<br/>ODE/SDE]
    end
    
    subgraph "External Models"
        K[Gemma3-4B-it<br/>Text Encoder]
        L[Jina CLIP v2<br/>Vision-Text]
        M[FLUX.1-dev VAE<br/>16ch]
    end
    
    A --> D
    A --> F
    A --> H
    A --> K
    A --> L
    A --> M
    
    B --> D
    B --> H
    B --> I
    B --> K
    B --> L
    B --> M
    
    C --> D
    C --> H
    C --> K
    C --> L
    C --> M
    
    D --> E
    F --> G
    H --> I
    H --> J
```

---

## 5. 주요 기능 모듈

### 5.1 모델 아키텍처
- **NextDiT**: 메인 Diffusion Transformer 모델 (3.5B 파라미터)
- **TimestepEmbedder**: 타임스텝을 벡터로 임베딩
- **JointAttention**: Flash Attention을 사용한 멀티헤드 어텐션
- **FeedForward**: SwiGLU 활성화 함수를 사용한 FFN
- **RMSNorm**: Root Mean Square Layer Normalization

### 5.2 텍스트 인코더
- **Gemma3-4B-it**: 주요 텍스트 인코더 (penultimate layer 사용)
- **Jina CLIP v2**: 풀링된 텍스트 특징 추출 및 AdaLN 조건화

### 5.3 데이터 처리
- **MyDataset**: 커스텀 데이터셋 클래스, 버킷 기반 배치 처리
- **T2IItemProcessor**: 텍스트-이미지 데이터 전처리
- **Aspect Ratio Bucketing**: 다양한 해상도 지원

### 5.4 Transport 시스템
- **Transport**: 학습 및 샘플링을 위한 전송 프레임워크
- **Sampler**: ODE/SDE 기반 샘플링
- **DPM-Solver**: 고속 샘플링을 위한 Diffusion Probabilistic Model Solver

### 5.5 분산 학습
- **FSDP**: Fully Sharded Data Parallel
- **Mixed Precision**: BF16/FP32 혼합 정밀도 학습
- **Gradient Clipping**: 그래디언트 정규화

---

## 6. 데이터 흐름

```mermaid
sequenceDiagram
    participant User
    participant Model as NextDiT Model
    participant TextEnc as Text Encoders
    participant VAE as FLUX VAE
    participant Transport
    
    User->>TextEnc: Input Prompt (XML/Natural/Tags)
    TextEnc->>TextEnc: Gemma3-4B-it encoding
    TextEnc->>TextEnc: Jina CLIP v2 pooling
    TextEnc-->>Model: Text embeddings + pooled features
    
    User->>VAE: Input Image (training)
    VAE->>VAE: Encode to 16ch latent
    VAE-->>Model: Latent representation
    
    Model->>Transport: Forward pass
    Transport->>Transport: Sample timestep
    Transport->>Transport: Add noise
    Transport-->>Model: Noisy latent + time
    
    Model->>Model: Predict noise/velocity
    Model-->>Transport: Prediction
    
    Transport->>Transport: Compute loss
    Transport-->>Model: Loss value
    
    alt Sampling Mode
        Model->>Transport: Generate from noise
        Transport->>Transport: ODE/SDE integration
        Transport-->>VAE: Clean latent
        VAE->>VAE: Decode to image
        VAE-->>User: Generated Image
    end
```

---

## 7. 주요 파일 설명

| 파일 | 설명 | 주요 클래스/함수 |
|------|------|------------------|
| `finetune.py` | 메인 학습 스크립트 | `main()`, `encode_prompt()`, `setup_fsdp_sync()` |
| `sample.py` | 샘플링 스크립트 | `main()`, `encode_prompt()` |
| `demo.py` | Gradio 데모 인터페이스 | `main()`, `model_main()`, `on_submit()` |
| `models/model.py` | NextDiT 모델 정의 | `NextDiT`, `JointAttention`, `FeedForward`, `TransformerBlock` |
| `models/components.py` | 컴포넌트 | `RMSNorm` |
| `data/dataset.py` | 데이터셋 클래스 | `MyDataset`, `ItemProcessor` |
| `data/data_reader.py` | 데이터 읽기 | `read_general()` |
| `transport/transport.py` | Transport 프레임워크 | `Transport`, `Sampler`, `ModelType`, `PathType` |
| `transport/dpm_solver.py` | DPM Solver | `DPM_Solver`, `NoiseScheduleFlow` |
| `transport/integrators.py` | ODE/SDE 적분기 | `ode()`, `sde()` |
| `imgproc.py` | 이미지 전처리 | `generate_crop_size_list()`, `center_crop()` |
| `parallel.py` | 분산 학습 유틸 | `distributed_init()`, `get_intra_node_process_group()` |

---

## 8. 학습 파이프라인 요약

```mermaid
graph TB
    subgraph "입력"
        A1[XML Structured Prompt]
        A2[Natural Language]
        A3[Tags]
        A4[Training Images]
    end
    
    subgraph "인코딩"
        B1[Gemma3-4B-it]
        B2[Jina CLIP v2]
        B3[FLUX VAE 16ch]
    end
    
    subgraph "모델"
        C1[NextDiT 3.5B]
        C2[TimestepEmbedder]
        C3[TransformerBlocks]
    end
    
    subgraph "학습"
        D1[Transport]
        D2[Loss Computation]
        D3[FSDP Backward]
        D4[AdamW Optimizer]
    end
    
    subgraph "출력"
        E1[Checkpoint]
        E2[EMA Model]
        E3[Logs \\u0026 Metrics]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    A1 --> B2
    A2 --> B2
    A3 --> B2
    A4 --> B3
    
    B1 --> C1
    B2 --> C1
    B3 --> C1
    C1 --> C2
    C2 --> C3
    
    C3 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> D4
    
    D4 --> E1
    D4 --> E2
    D4 --> E3
```

---

## 9. 시스템 요구사항

- **GPU**: CUDA 지원 GPU (최소 24GB VRAM 권장)
- **Python**: 3.8+
- **주요 라이브러리**:
  - PyTorch 2.0+
  - Transformers
  - Diffusers
  - Flash Attention (선택 사항, 성능 향상)
  - FSDP (분산 학습)

---

## 10. 확장 가능성

1. **LoRA 파인튜닝**: [NewbieLoraTrainer](https://github.com/NewBieAI-Lab/NewbieLoraTrainer)
2. **ComfyUI 통합**: [ComfyUI-Newbie-V0.1](https://github.com/NewBieAI-Lab/ComfyUI-Newbie-V0.1)
3. **커스텀 데이터셋**: XML 구조화 프롬프트 지원
4. **멀티 해상도**: Aspect ratio bucketing

---

이 문서는 NewBie image Exp0.1 프로젝트의 전체 아키텍처를 시각화하고 설명합니다.
