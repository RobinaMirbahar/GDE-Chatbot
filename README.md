# GDE-Chatbot

> **Author & Architect:** [Robina Mirbahar](https://github.com/RobinaMirbahar) ([Google Developer Expert in Cloud & AI](https://developers.google.com/profile/u/106918534005910641104?utm_campaign=deveco_gdemembers&utm_source=deveco) · Google Cloud Champion Innovator · Co-Author of [*AI Engineering*](https://www.amazon.com/dp/B0GWCZWCF4))  
> 📚 **Architecture Track:** Google Cloud & Gemini AI Community Knowledge Assistant  
> **Core Stack:** [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai/docs?utm_campaign=deveco_gdemembers&utm_source=deveco), [Google Gemini API](https://ai.google.dev/gemini-api/docs?utm_campaign=deveco_gdemembers&utm_source=deveco), Grounded RAG Retrieval, and Python.

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
```

---
---

### Official Google Developer Resources
- [Google Gemini API & AI Studio Docs](https://ai.google.dev/gemini-api/docs?utm_campaign=deveco_gdemembers&utm_source=deveco)
- [Google Cloud Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs?utm_campaign=deveco_gdemembers&utm_source=deveco)
