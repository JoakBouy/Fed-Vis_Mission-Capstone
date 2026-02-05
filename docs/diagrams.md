# Fed-Vis System Design Diagrams

All diagrams use Mermaid syntax for rendering. Copy into any Mermaid-compatible viewer or include directly in markdown documents.

---

## 1. System Architecture Overview

```mermaid
flowchart TB
    subgraph Privacy["Privacy Layer - Data Never Leaves Hospital"]
        subgraph H1["Kenyatta National Hospital"]
            D1[(Brain MRI Data)]
            M1[Local Model]
        end
        subgraph H2["Moi Teaching Hospital"]
            D2[(Lung CT Data)]
            M2[Local Model]
        end
        subgraph H3["Aga Khan Hospital"]
            D3[(Prostate MRI Data)]
            M3[Local Model]
        end
    end

    subgraph Federation["Federation Layer"]
        FS[Flower Aggregation Server<br/>FedAvg Strategy]
    end

    subgraph Inference["Inference Layer"]
        API[FastAPI Service]
        MC[Marching Cubes<br/>Mesh Generation]
    end

    subgraph Visualization["Visualization Layer"]
        UI[React + Three.js<br/>Doctor's Cockpit]
    end

    M1 -->|Weights Only| FS
    M2 -->|Weights Only| FS
    M3 -->|Weights Only| FS
    FS -->|Global Model| API
    API --> MC
    MC -->|3D Mesh| UI

    style Privacy fill:#1a365d,color:#fff
    style Federation fill:#2c5282,color:#fff
    style Inference fill:#2b6cb0,color:#fff
    style Visualization fill:#3182ce,color:#fff
```

---

## 2. Attention U-Net Architecture

```mermaid
flowchart TB
    subgraph Encoder["Encoder Path"]
        E1["Conv Block 1<br/>32 channels"]
        E2["Conv Block 2<br/>64 channels"]
        E3["Conv Block 3<br/>128 channels"]
        E4["Conv Block 4<br/>256 channels"]
    end

    subgraph Bottleneck["Bottleneck"]
        BN["Conv Block<br/>512 channels"]
    end

    subgraph Decoder["Decoder Path"]
        D4["UpConv Block 4<br/>256 channels"]
        D3["UpConv Block 3<br/>128 channels"]
        D2["UpConv Block 2<br/>64 channels"]
        D1["UpConv Block 1<br/>32 channels"]
    end

    subgraph Attention["Attention Gates"]
        AG4{{"AG4"}}
        AG3{{"AG3"}}
        AG2{{"AG2"}}
        AG1{{"AG1"}}
    end

    Input["Input Volume<br/>(1, 64, 128, 128)"] --> E1
    E1 -->|MaxPool| E2
    E2 -->|MaxPool| E3
    E3 -->|MaxPool| E4
    E4 -->|MaxPool| BN

    BN -->|UpConv| D4
    E4 --> AG4
    D4 --> AG4
    AG4 --> D4

    D4 -->|UpConv| D3
    E3 --> AG3
    D3 --> AG3
    AG3 --> D3

    D3 -->|UpConv| D2
    E2 --> AG2
    D2 --> AG2
    AG2 --> D2

    D2 -->|UpConv| D1
    E1 --> AG1
    D1 --> AG1
    AG1 --> D1

    D1 --> Output["Output Mask<br/>(1, 64, 128, 128)"]

    style Encoder fill:#3182ce,color:#fff
    style Decoder fill:#38a169,color:#fff
    style Bottleneck fill:#805ad5,color:#fff
    style Attention fill:#dd6b20,color:#fff
```

---

## 3. Federated Learning Flow

```mermaid
sequenceDiagram
    participant KNH as Kenyatta Hospital<br/>(Brain)
    participant MTH as Moi Teaching<br/>(Lung)
    participant AKH as Aga Khan<br/>(Prostate)
    participant Server as Flower Server

    Note over KNH,AKH: Round 1 - Local Training
    KNH->>KNH: Train on local brain MRI
    MTH->>MTH: Train on local lung CT
    AKH->>AKH: Train on local prostate MRI

    Note over KNH,Server: Upload Weights Only (No Data)
    KNH-->>Server: Model weights Δw₁
    MTH-->>Server: Model weights Δw₂
    AKH-->>Server: Model weights Δw₃

    Note over Server: FedAvg Aggregation
    Server->>Server: w_global = Σ(nᵢ/n)wᵢ

    Note over Server,AKH: Broadcast Global Model
    Server-->>KNH: Updated global model
    Server-->>MTH: Updated global model
    Server-->>AKH: Updated global model

    Note over KNH,AKH: Repeat for 20 rounds
```

---

## 4. Data Harmonization Pipeline

```mermaid
flowchart LR
    subgraph Raw["Raw Data"]
        B[BraTS<br/>4-channel MRI]
        P[Prostate<br/>Multi-site MRI]
        L[Lung CT<br/>DICOM/PNG]
    end

    subgraph Harmonize["Harmonization"]
        H1[Channel Selection<br/>FLAIR only]
        H2[Resample to<br/>1mm³ isotropic]
        H3[Z-Score<br/>Normalization]
        H4[Crop/Pad to<br/>64×128×128]
    end

    subgraph Output["Unified Format"]
        O1[Node 1<br/>Brain .npy]
        O2[Node 2<br/>Prostate .npy]
        O3[Node 3<br/>Lung .npy]
    end

    B --> H1
    P --> H1
    L --> H1
    H1 --> H2 --> H3 --> H4
    H4 --> O1
    H4 --> O2
    H4 --> O3

    style Raw fill:#e53e3e,color:#fff
    style Harmonize fill:#dd6b20,color:#fff
    style Output fill:#38a169,color:#fff
```

---

## 5. ERD Diagram

```mermaid
erDiagram
    HOSPITAL ||--o{ PATIENT : has
    PATIENT ||--o{ VOLUME : has
    VOLUME ||--|| SEGMENTATION : produces
    VOLUME }o--|| MODALITY : uses
    HOSPITAL ||--|| FED_CLIENT : runs

    HOSPITAL {
        int hospital_id PK
        string name
        string location
        string organ_specialty
    }

    PATIENT {
        int patient_id PK
        int hospital_id FK
        string anonymized_id
        date scan_date
    }

    VOLUME {
        int volume_id PK
        int patient_id FK
        int modality_id FK
        string file_path
        float spacing_d
        float spacing_h
        float spacing_w
    }

    MODALITY {
        int modality_id PK
        string name
        string organ
    }

    SEGMENTATION {
        int seg_id PK
        int volume_id FK
        float dice_score
        string mesh_path
        blob attention_map
    }

    FED_CLIENT {
        int client_id PK
        int hospital_id FK
        int samples_count
        int last_round
    }
```

---

## 6. Deployment Architecture (Docker)

```mermaid
flowchart TB
    subgraph Docker["Docker Compose"]
        subgraph FL["Federated Learning"]
            C1[fedvis-client-brain]
            C2[fedvis-client-lung]
            C3[fedvis-client-prostate]
            S[fedvis-server]
        end

        subgraph Backend["Backend Services"]
            API[fedvis-api<br/>FastAPI]
            DB[(PostgreSQL)]
        end

        subgraph Frontend["Frontend"]
            WEB[fedvis-web<br/>React + Three.js]
            NGINX[nginx<br/>Reverse Proxy]
        end
    end

    C1 --> S
    C2 --> S
    C3 --> S
    S --> API
    API --> DB
    API --> WEB
    NGINX --> WEB
    NGINX --> API

    Internet((Internet)) --> NGINX
```

---

## 7. Gantt Chart Timeline

```mermaid
gantt
    title Fed-Vis 14-Week Development Timeline
    dateFormat  YYYY-MM-DD
    section Sprint 1
    Foundation + DVC           :s1, 2026-01-27, 14d
    Data Harmonization         :s1b, 2026-01-27, 14d
    section Sprint 2
    Attention U-Net            :s2, after s1, 14d
    Unit Tests                 :s2b, after s1, 14d
    section Sprint 3
    Centralized Training       :s3, after s2, 14d
    MLflow Logging             :s3b, after s2, 14d
    section Sprint 4
    Flower Integration         :s4, after s3, 14d
    3-Client Simulation        :s4b, after s3, 14d
    section Sprint 5
    FastAPI Service            :s5, after s4, 14d
    Docker Containerization    :s5b, after s4, 14d
    section Sprint 6
    React + Three.js           :s6, after s5, 14d
    Doctor's Cockpit           :s6b, after s5, 14d
    section Sprint 7
    Documentation              :s7, after s6, 14d
    Final Testing              :s7b, after s6, 14d
```
