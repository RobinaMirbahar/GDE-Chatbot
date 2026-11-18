# GDE-Chatbot

> **Author & Architect:** [Robina Mirbahar](https://github.com/RobinaMirbahar) ([Google Developer Expert in Cloud & AI](https://developers.google.com/profile/u/106918534005910641104?utm_campaign=deveco_gdemembers&utm_source=deveco) · Google Cloud Champion Innovator · Co-Author of [*AI Engineering*](https://www.amazon.com/dp/B0GWCZWCF4))  
> **Focus:** Google Cloud & Gemini AI Community Assistant Built with Python, Vertex AI & Grounded Knowledge Retrieval

---

## 🎯 Why I Built GDE Community Assistant Chatbot (Engineering Purpose & Real-World Value)

### 1. 🤝 Scaling Developer Mentorship Across Community Events
As a Google Developer Expert (GDE), I frequently answer hundreds of recurring questions from students and builders about Google Cloud architecture, Vertex AI quotas, and learning paths. I built GDE-Chatbot as a grounded community assistant.

### 2. 📚 Context-Aware Multi-Turn Guidance
I engineered the session loop to preserve conversation history and provide runnable `gcloud` and Python SDK snippets tailored to the developer's experience level.

### 3. ⚡ Lightweight & Extensible Python Blueprint
The codebase is intentionally clean and modular so local GDG and WTM chapters can fork and deploy their own event assistant in minutes.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Dev ["1. Community Member Inquiry"]
        D_Q["Cloud Architecture / Vertex AI / Career Question"]
    end
    subgraph Bot ["2. GDE Assistant Core (Python + Gemini)"]
        B_Mem["Multi-Turn Session Context"]
        B_Prompt["GDE Mentorship System Instructions"]
        B_LLM["Google Gemini / Vertex AI Inference"]
    end
    subgraph Ans ["3. Grounded Technical Response"]
        A_Code["Runnable SDK Snippets & Official Doc Links"]
    end
    D_Q --> B_Mem --> B_Prompt --> B_LLM --> A_Code
    style Dev fill:#fce7f3,stroke:#db2777,stroke-width:2px,color:#831843
    style D_Q fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    style Bot fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    style Core fill:#f3e8ff,stroke:#9333ea,stroke-width:2px,color:#581c87
    style B_Mem fill:#fce7f3,stroke:#db2777,stroke-width:2px,color:#831843
    style B_Prompt fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    style B_LLM fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    style Ans fill:#f3e8ff,stroke:#9333ea,stroke-width:2px,color:#581c87
    style A_Code fill:#fce7f3,stroke:#db2777,stroke-width:2px,color:#831843
```

---
---

### Official Google Developer Resources
- [Google Gemini API & AI Studio Docs](https://ai.google.dev/gemini-api/docs?utm_campaign=deveco_gdemembers&utm_source=deveco)
- [Google Cloud Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs?utm_campaign=deveco_gdemembers&utm_source=deveco)
