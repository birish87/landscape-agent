# Production Checklist
## Steps to deploy this project for real

Work through these in order. Each step is independent.

---

## Phase 1: Core Functionality (Week 1-2)

- [ ] Replace `dummy_db.json` crews with real SingleOps API calls
- [ ] Replace `dummy_db.json` pricing with LMN service template exports
- [ ] Seed ChromaDB with real knowledge documents (OSU Extension guides + your SOPs)
- [ ] Switch `retrieve_knowledge_simple()` to `retrieve_knowledge_semantic()` (in seed_vector_db.py)
- [ ] Add input validation (reject empty messages, very long inputs, etc.)

## Phase 2: API & Infrastructure (Week 2-3)

- [ ] Wrap `run_pipeline()` in a FastAPI endpoint: `POST /api/v1/chat`
- [ ] Add authentication (API key or JWT)
- [ ] Move secrets to AWS Secrets Manager or Azure Key Vault
- [ ] Add structured logging (send token usage + latency to Datadog)
- [ ] Containerize with Docker

## Phase 3: Reliability (Week 3-4)

- [ ] Add response caching (Redis) — same query + same context = skip API call
- [ ] Implement conversation history (pass last N messages for multi-turn chat)
- [ ] Add evaluation pipeline: human-rate 50 responses, track quality over time
- [ ] Set up alerts for: high error rate, high latency, high cost

## Phase 4: Advanced Features (Month 2)

- [ ] Upgrade vector DB from ChromaDB to Pinecone (3 line change)
- [ ] Add LaunchDarkly AI Configs for prompt A/B testing
- [ ] Connect Snowflake Cortex for job demand forecasting
- [ ] Add Claude vision API to site_analysis_agent.py for photo analysis
- [ ] Build .NET/C# version using Semantic Kernel (same patterns, different syntax)

## Phase 5: Production Scale (Month 3+)

- [ ] Move orchestrator to async (asyncio.gather for parallel tool calls)
- [ ] Add streaming responses (Anthropic SDK supports stream=True)
- [ ] Build admin dashboard showing agent traces, costs, and quality metrics
- [ ] Integrate with SingleOps webhooks (auto-trigger agent on new job created)
