# phases.md

> Breakdown of CareerCopilot AI into manageable build phases
> Each phase lists the exact files to create (backend + frontend), matching `architecture.md`.

---

## PHASE 1 — Project Setup & Resume Ingestion (MVP)

**Goal:** Upload a resume PDF, extract raw text, store it in the DB.

**Backend files:**
```
backend/
├── requirements.txt
├── .env
├── .env.example
└── app/
    ├── __init__.py
    ├── main.py                    # FastAPI app instance, includes routers
    ├── core/
    │   ├── __init__.py
    │   └── config.py              # pydantic-settings: env vars, DB URL, etc.
    ├── api/
    │   ├── __init__.py
    │   └── resume.py              # POST /api/v1/resume/upload
    ├── services/
    │   ├── __init__.py
    │   └── resume_parser.py       # PDF text extraction (pypdf)
    ├── models/
    │   ├── __init__.py
    │   └── schemas.py             # Pydantic: ResumeUploadResponse
    └── db/
        ├── __init__.py
        ├── database.py            # SQLAlchemy engine + session
        └── models.py              # ORM: Resume table (id, filename, raw_text, uploaded_at)
```
```
backend/tests/
└── test_resume_upload.py
```

**Frontend files:**
```
frontend/src/
├── App.jsx
├── main.jsx
├── pages/
│   └── Upload.jsx                 # Upload form (resume file input)
└── api/
    ├── client.js                  # axios instance, base URL
    └── resume.js                  # uploadResume() call
```

---

## PHASE 2 — Structured Extraction

**Goal:** Turn raw resume text into structured JSON (skills, experience, education) via LLM.

**Backend files (new/updated):**
```
app/
├── prompts/
│   ├── __init__.py
│   └── resume_extraction_prompt.py   # prompt template for extraction
├── services/
│   └── llm_service.py                # centralized LLM calls + retry logic
├── models/
│   └── schemas.py                    # (updated) add ResumeParsed schema
└── api/
    └── resume.py                     # (updated) add POST /api/v1/resume/parse
```
```
backend/tests/
└── test_resume_parse.py
```

**Frontend files (new):**
```
frontend/src/
├── components/
│   └── ResumePreviewCard.jsx      # shows parsed skills/education/experience
└── pages/
    └── Upload.jsx                 # (updated) show preview after parsing
```

---

## PHASE 3 — RAG System

**Goal:** Chunk + embed resume/JD text, retrieve relevant chunks, answer questions via chat.

**Backend files (new):**
```
app/
├── vectorstore/
│   ├── __init__.py
│   └── chroma_client.py           # Chroma init + collection helpers
├── services/
│   ├── embedding_service.py       # text → embeddings
│   └── rag_service.py             # chunking + retrieval + context assembly
├── api/
│   └── chat.py                    # POST /api/v1/chat
└── models/
    └── schemas.py                 # (updated) ChatRequest, ChatResponse, SourceChunk
```
```
backend/tests/
└── test_chat.py
```

**Frontend files (new):**
```
frontend/src/
├── pages/
│   └── Chat.jsx                   # chat UI
├── components/
│   ├── ChatBubble.jsx
│   └── SourceCitation.jsx
├── hooks/
│   └── useChat.js                 # chat state management
└── api/
    └── chat.js                    # sendMessage() call
```

---

## PHASE 4 — Job Matching & Skill Gap Analysis

**Goal:** Compare resume skills vs job description, compute match score.

**Backend files (new):**
```
app/
├── api/
│   └── jobs.py                    # POST /api/v1/jobs/analyze
├── services/
│   └── matching_service.py        # skill comparison + score calculation
└── models/
    └── schemas.py                 # (updated) JobDescriptionInput, MatchResult
```
```
backend/tests/
└── test_jobs.py
```

**Frontend files (new):**
```
frontend/src/
├── pages/
│   └── MatchReport.jsx            # score ring + skill badges
├── components/
│   ├── ScoreRing.jsx
│   └── SkillBadge.jsx             # matched (green) / missing (amber)
└── api/
    └── jobs.js                    # analyzeJob() call
```

---

## PHASE 5 — AI Agent with Tool Calling

**Goal:** Multi-step agent that orchestrates resume, matching, and RAG tools to build a roadmap.

**Backend files (new):**
```
app/
├── agents/
│   ├── __init__.py
│   ├── career_agent.py            # LangGraph agent graph
│   └── tools.py                   # get_resume(), analyze_skills(), get_job_requirements(),
│                                   # search_vector_database(), calculate_match_score(),
│                                   # create_learning_plan()
└── api/
    └── agent.py                   # POST /api/v1/agent/roadmap
```
```
backend/tests/
└── test_agent.py
```

**Frontend files (new):**
```
frontend/src/
├── pages/
│   └── Roadmap.jsx                # agent-generated learning plan
├── components/
│   └── RoadmapChecklist.jsx
└── api/
    └── agent.js                   # getRoadmap() call
```

---

## PHASE 6 — Evaluation, Monitoring, Testing & Deployment

**Goal:** Measure RAG quality, monitor production, containerize, deploy.

**Backend files (new):**
```
evaluation/
├── dataset.py                     # question/expected-answer test set
└── run_ragas.py                   # RAGAS metrics: faithfulness, relevancy, precision/recall

app/
├── api/
│   └── evaluation.py              # GET /api/v1/evaluation/metrics
├── services/
│   └── monitoring_service.py      # request count, latency, error tracking
└── middleware/
    └── logging_middleware.py      # request/response logging

backend/
└── Dockerfile
```
```
backend/tests/
├── test_evaluation.py
└── (finish integration coverage for all endpoints)
```

**Frontend files (new):**
```
frontend/
├── src/pages/
│   └── Dashboard.jsx              # admin monitoring view
└── Dockerfile
```

**Project root files (new):**
```
ai-career-intelligence/
└── docker-compose.yml             # backend + frontend + postgres + vector db
```

**Deployment:**
- Frontend → Vercel
- Backend → Railway / Render
- Vector DB → Qdrant Cloud

---

## Notes
- Do not start Phase 3 (RAG) before Phase 2 (structured extraction) is stable — RAG depends on clean parsed data.
- Phase 5 (Agent) is optional-but-recommended for portfolio strength; can be deprioritized if time-constrained.
- Re-run RAGAS evaluation (Phase 6) after any change to chunking, embedding model, or prompt — treat it as a regression test.
- Every `services/` file gets a matching `tests/` file in the same phase it's introduced — don't defer tests to Phase 6.
