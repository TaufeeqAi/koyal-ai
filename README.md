# KoyalAI

**Multilingual Voice AI for Indian Enterprises**

Production-grade voice agents that speak Hindi, English, Hinglish, and 10+ Indian languages. Built for BFSI, e-commerce, and healthcare — with per-tenant cost tracking, real-time observability, and enterprise-grade safety.

---

## What It Does

KoyalAI handles inbound customer service calls and runs outbound campaigns in the caller's language — automatically detecting and switching between Hindi, English, and Hinglish mid-conversation. It retrieves answers from tenant-specific knowledge bases, escalates emergencies, and tracks every rupee spent per tenant in real time.

**Live Capabilities:**
- **Inbound Voice AI** — Answers customer queries in Hindi, English, or Hinglish with Indian-accented TTS
- **Outbound Campaigns** — Automated EMI reminders, appointment notifications, order updates
- **Real-Time Language Detection** — Identifies language from speech and switches response language instantly
- **Per-Tenant Cost Tracking** — STT/TTS/LLM costs tracked in ₹ per tenant via Redis
- **Live Observability** — Grafana dashboards, Prometheus metrics, and Langfuse LLM traces
- **Safety Escalation** — Hindi + English emergency keyword detection with semantic similarity fallback

---

## Architecture

```
Caller (Hindi/English/Hinglish)
         │
         ▼
┌─────────────────────┐
│    LiveKit SIP      │  ← Telephony (inbound + outbound)
│    Server           │    Indian number simulation
└──────────┬──────────┘
           │ Audio stream (PCM)
           ▼
┌─────────────────────┐
│   Language          │  ← Detects: Hindi / English / Hinglish
│   Detector          │    Routes to correct ASR model
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Sarvam Saaras V3  │  ← STT: Indian-optimised ASR
│   STT Engine        │    Handles code-switching natively
└──────────┬──────────┘
           │ Transcript + detected language
           ▼
┌──────────────────────────────────────────────────────┐
│              LangGraph Agent Pipeline                │
│                                                      │
│  ┌─────────────┐   ESCALATE   ┌──────────────────┐   │
│  │ Safety Gate │ ───────────► │ Escalation       │   │
│  │ (Hindi +    │              │ Handler          │   │
│  │  English)   │              │ (hardcoded)      │   │
│  └──────┬──────┘              └──────────────────┘   │
│         │ SAFE                                       │
│         ▼                                            │
│  ┌─────────────┐                                     │
│  │  Language   │  ← Translates query to English      │
│  │  Bridge     │    for LLM reasoning                │
│  └──────┬──────┘                                     │
│         ▼                                            │
│  ┌─────────────┐                                     │
│  │  Retrieval  │  ← Qdrant (tenant-isolated)         │
│  │  Agent      │    Hindi + English chunks           │
│  └──────┬──────┘                                     │
│         ▼                                            │
│  ┌─────────────┐                                     │
│  │  Response   │  ← Groq Llama 3.3, temp=0           │
│  │  Agent      │    Responds in caller's language    │
│  └──────┬──────┘                                     │
│         ▼                                            │
│  ┌─────────────┐                                     │
│  │Verification │  ← Chain of Verification            │
│  │  Agent      │    Language-aware faithfulness      │
│  └──────┬──────┘                                     │
└─────────┼────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│  Sarvam Bulbul V3   │  ← TTS: 25+ Indian voices
│  TTS Engine         │    Matches caller's language
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Langfuse           │     │  Prometheus +       │
│  LLM Observability  │     │  Grafana            │
│  (per trace)        │     │  (infra metrics)    │
└─────────────────────┘     └─────────────────────┘
          │
          ▼
    Caller hears response in their language
```

---

## Multi-Tenancy + Cost Tracking

Each enterprise tenant operates in complete isolation with dedicated vector collections and real-time cost accounting.

**Vector Store Isolation:**

```
Qdrant
├── Collection: tenant_hdfc_bank
│   ├── Hindi loan guidelines
│   ├── EMI schedule protocols
│   └── Recovery scripts (Hindi + English)
│
└── Collection: tenant_swiggy_support
    ├── Order tracking scripts
    ├── Refund protocols
    └── Escalation guidelines
```

**Real-Time Cost Tracking (Redis):**

```
tenant_hdfc_bank:
├── stt_minutes: 142.3
├── tts_chars: 48291
├── llm_tokens: 128400
└── total_cost_inr: ₹89.40

tenant_swiggy_support:
├── stt_minutes: 89.1
├── tts_chars: 31200
└── total_cost_inr: ₹54.20
```

---

## Stack

| Layer | Tool | Rationale |
|-------|------|-----------|
| **STT** | Sarvam Saaras V3 | Indian-optimised, telephony-trained, handles Hinglish natively |
| **TTS** | Sarvam Bulbul V3 | 25+ Indian voices (meera, anushka, madhura, pavithra, hema, gagan, bani) |
| **Translation** | Sarvam Mayura | English ↔ Hindi formal translation for LLM reasoning bridge |
| **Language Detection** | Sarvam LID + Script Analysis | Two-layer: Devanagari/Latin script check (instant) + API confirmation |
| **LLM** | Groq Llama 3.3-70B | Fast inference, free tier, strong reasoning in English |
| **Embeddings** | LaBSE (768-dim) | 109 languages, same semantic space across Hindi/English — "EMI" and "ईएमआई" map closely |
| **Vector DB** | Qdrant | Tenant-isolated collections, cosine similarity, open source |
| **Orchestration** | LangGraph | Multi-agent pipeline with conditional edges for escalation |
| **Telephony** | LiveKit SIP | WebRTC rooms + SIP trunking for inbound/outbound PSTN |
| **Cache/Cost** | Redis | Sub-millisecond writes for live call cost accumulation |
| **LLM Observability** | Langfuse | Per-trace logging, span tracking, prompt versioning |
| **Infra Metrics** | Prometheus + Grafana | Real-time dashboards with ₹/minute panels |
| **Frontend** | Next.js 14 + Tailwind | Live call monitor, tenant onboarding, campaign manager |

---

## Why LaBSE Over MiniLM

LaBSE is chosen specifically for Indian multilingual requirements:

| Factor | MiniLM | LaBSE |
|--------|--------|-------|
| Languages | English-optimised | 109 languages including all major Indian languages |
| Hindi similarity | Degrades significantly | Native quality |
| Code-mixed text | Fails on Hinglish | Handles Hinglish natively |
| Cross-lingual | No | "EMI" ≈ "ईएमआई" in same vector space |
| Dimension | 384 | 768 |
| Tradeoff | Faster, smaller | 2x storage, worth it for accuracy |

---

## Language Detection

Two-layer detection for speed and precision:

**Layer 1 — Script Analysis (instant, deterministic):**
- Devanagari characters (`\u0900-\u097F`) detected → Hindi present
- Latin characters (`a-zA-Z`) detected → English present
- Both → Hinglish (code-mixed)

**Layer 2 — Sarvam LID API (confirmation):**
- Handles ambiguous cases
- Detects specific regional Indian languages (Marathi, Tamil, Telugu, Kannada, Bengali)
- 3-second timeout with fallback to Layer 1

---

## Safety & Emergency Handling

**Keyword Detection (Layer 1):**
- Hindi emergencies: `दिल का दौरा`, `साँस नहीं आ रही`, `आत्महत्या`, `मरना चाहता`
- English emergencies: `chest pain`, `heart attack`, `suicidal`, `can't breathe`
- Hinglish emergencies: `dil mein dard`, `sans nahi aa raha`, `marna chahta`
- Banking emergencies: `खाता खाली हो गया`, `fraud ho gaya`, `account hack`, `unauthorized transaction`

**Semantic Detection (Layer 2):**
- LaBSE embeddings of reference emergency sentences
- Cosine similarity threshold ≥ 0.80 catches paraphrases and misspellings

**Escalation Response:**
- Language-matched hardcoded message (e.g., Hindi emergency → Hindi escalation)
- Immediate human handoff via LiveKit SIP transfer

---

## Frontend Dashboard

**Live Call Monitor (`/`)**
- Real-time bilingual transcript with language badges (हिंदी / English / Hinglish)
- WebSocket-connected to backend for instant turn updates
- Connection status indicator

**Tenant Onboarding (`/tenants`)**
- Create tenant with primary language selection
- Upload knowledge documents (`.txt`, `.pdf`) with language tagging
- Auto-ingestion into isolated Qdrant collection

**Outbound Campaign Manager (`/outbound`)**
- Launch campaigns with personalised scripts (`{name}` substitution)
- Async semaphore-controlled concurrency (default 5 simultaneous calls)
- Live status tracking per contact

**Evaluation Dashboard (`/evals`)**
- RAGAS faithfulness, answer relevancy, context precision, context recall
- Per-language score breakdown with threshold indicators (green ≥ 82%, yellow ≥ 70%, red < 70%)
- Auto-refresh every 30 seconds

---

## Observability

**Prometheus Metrics Exported:**
- `koyal_calls_total` — by tenant, language, call type, outcome
- `koyal_call_duration_seconds` — histogram with 10s–600s buckets
- `koyal_active_calls` — gauge per tenant
- `koyal_stt_latency_ms` / `koyal_llm_latency_ms` / `koyal_tts_latency_ms` — latency histograms
- `koyal_ttfr_ms` — time to first response
- `koyal_emergency_escalations_total` — by tenant, language, reason
- `koyal_cost_inr_total` — by tenant, cost type
- `koyal_ragas_faithfulness` — gauge per tenant/language
- `koyal_retrieval_relevance_score` — histogram

**Grafana Dashboard Panels:**
- Active calls by tenant (live)
- ₹/minute burn rate per tenant
- STT/TTS/LLM latency percentiles (p50, p95, p99)
- Language distribution pie chart
- Emergency escalation frequency
- RAGAS score trends over time

**Langfuse Tracing:**
- Every agent node logged as a span
- Full prompt/response capture with versioning
- Retrieval context attached to generation spans
- Cost attribution per trace

---

## Evaluation Suite

**RAGAS Multilingual Evaluation:**
- Test cases in Hindi, English, and Hinglish
- Metrics: faithfulness, answer relevancy, context precision, context recall
- Assertion: faithfulness ≥ 0.80 for all languages
- Sample queries:
  - Hindi: *"मेरी EMI कब कटती है?"* → expects *"5 तारीख को"*
  - English: *"When is EMI deducted?"* → expects *"5th of every month"*
  - Hinglish: *"EMI miss ho gayi, kya penalty hai?"* → expects *"500"*

**DeepEval Safety Tests:**
- Hindi emergency keyword detection
- English emergency keyword detection
- Hinglish emergency keyword detection
- Semantic similarity paraphrase detection
- Banking fraud escalation triggers

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/TaufeeqAi/koyal-ai
cd koyal-ai
cp .env.example .env
# Edit .env: add SARVAM_API_KEY and GROQ_API_KEY

# 2. Start all services
docker-compose up -d

# 3. Ingest tenant knowledge bases
python scripts/ingest_all.py

# 4. Run tests
pytest tests/ -v

# 5. Run evaluation suite
python scripts/run_evals.py

# 6. Open dashboard
open http://localhost:3000
```

**Services:**
| Service | URL | Credentials |
|---------|-----|-------------|
| Frontend | http://localhost:3000 | — |
| Backend API | http://localhost:8000 | — |
| Langfuse | http://localhost:3001 | auto-configured |
| Grafana | http://localhost:3002 | admin / koyal2025 |
| Prometheus | http://localhost:9090 | — |
| LiveKit | ws://localhost:7880 | devkey / secret |

---

## Configuration

**`.env` (required):**
```env
# Sarvam AI (free tier — signup at app.sarvam.ai)
SARVAM_API_KEY=your_sarvam_api_key

# Groq (free tier — console.groq.com)
GROQ_API_KEY=your_groq_api_key

# Local infrastructure
QDRANT_HOST=localhost
QDRANT_PORT=6333
REDIS_HOST=localhost
REDIS_PORT=6379

# Langfuse (self-hosted)
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=http://localhost:3001

# LiveKit
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
LIVEKIT_WS_URL=ws://localhost:7880
LIVEKIT_SIP_TRUNK_ID=trunk_in_1

# App
APP_ENV=development
DEFAULT_LANGUAGE=hi-IN
SUPPORTED_LANGUAGES=hi-IN,en-IN,mr-IN,ta-IN,te-IN,kn-IN,bn-IN
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/telephony/livekit/webhook` | POST | LiveKit room event webhooks |
| `/telephony/token` | POST | Issue participant token for browser client |
| `/tenant/costs` | GET | Per-tenant cost breakdown (₹) |
| `/calls/active` | GET | List active call sessions |
| `/evals/ragas` | GET | RAGAS scores by tenant |
| `/outbound/campaign` | POST | Launch outbound campaign |
| `/tenants/create` | POST | Create new tenant |
| `/documents/upload` | POST | Upload and ingest document |
| `/metrics` | GET | Prometheus metrics endpoint |
| `/health` | GET | Liveness/readiness probe |
| `/ws/transcript/{session_id}` | WS | Live transcript WebSocket |

---

## Resilience Patterns

| Component | Pattern | Implementation |
|-----------|---------|----------------|
| Sarvam STT | Timeout + Fallback | 10s timeout, return empty transcript on failure |
| Sarvam TTS | Chunking + Retry | 500-char chunks, 3 retries per chunk with backoff |
| Sarvam Translate | Timeout + Passthrough | 5s timeout, return original text on failure |
| Groq LLM | Retry + Fallback | 3 retries with exponential backoff, temperature=0 for determinism |
| Qdrant | Connection Pool | Persistent client with health check |
| Redis | Connection Retry | Auto-reconnect with 3 attempts |
| LiveKit | Room Reconnect | Automatic room reconnection on disconnect |

---

## Security

- **Secret Management:** All API keys via environment variables
- **Tenant Isolation:** Strict collection naming (`koyal_{tenant_id}`); cross-tenant query rejection at retriever level
- **Input Sanitisation:** Emergency keyword detection prevents prompt injection via safety gate
- **No Hardcoded Secrets:** All credentials externalised; `.env` gitignored
- **SIP Security:** LiveKit token-based room access; SIP trunk allow-lists configurable

---

## Cost Model

| Service | Rate | Free Tier |
|---------|------|-----------|
| Sarvam STT | ₹0.50/minute | ₹1,000 credits (~33 hours) |
| Sarvam TTS | ₹0.0015/character | Included in credits |
| Sarvam Translate | Included in STT/TTS | Included |
| Groq LLM | ₹0.10/1K tokens | Free tier (₹0) |
| Qdrant | ₹0 (self-hosted) | Open source |
| LiveKit | ₹0 (self-hosted) | Open source |
| Langfuse | ₹0 (self-hosted) | Open source |

---

## License

MIT — 100% open source.

---

## Built By

**Taufeeq Ahmad** — AI systems architect specialising in production voice AI for multilingual markets.

*Stack: Sarvam AI · LangGraph · Qdrant · Groq · LiveKit · Langfuse · Prometheus · Grafana · Next.js · Docker*