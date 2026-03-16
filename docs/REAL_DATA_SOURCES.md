# Real Data Sources
## How to Replace Dummy Data with Production Data

---

## 1. Crew & Scheduling Data → SingleOps API

SingleOps (a Granum product) has a REST API for crew management.

**Endpoint**: `GET /api/v1/crews`
**Auth**: Bearer token in Authorization header

```python
import requests

def get_crews_from_singleops():
    response = requests.get(
        "https://api.singleops.com/v1/crews",
        headers={"Authorization": f"Bearer {os.environ['SINGLEOPS_API_KEY']}"}
    )
    crews = response.json()
    return crews

def get_crew_availability(date: str):
    response = requests.get(
        f"https://api.singleops.com/v1/crews/available?date={date}",
        headers={"Authorization": f"Bearer {os.environ['SINGLEOPS_API_KEY']}"}
    )
    return response.json()
```

Replace `check_crew_availability()` in tools/tools.py with this function.

---

## 2. Pricing Data → LMN Service Templates

LMN (another Granum product) manages service pricing and templates.

**Option A**: Export from LMN UI
- Go to Services → Export CSV
- Parse the CSV and load into your pricing dict
- Schedule a nightly export to keep prices current

**Option B**: LMN API (if available on your plan)
```python
def get_service_pricing(service_type: str):
    response = requests.get(
        f"https://api.golmn.com/v1/services/{service_type}/pricing",
        headers={"Authorization": f"Bearer {os.environ['LMN_API_KEY']}"}
    )
    return response.json()
```

---

## 3. Knowledge Base Documents

These are the documents you embed into ChromaDB for RAG.

### Free Sources:
| Source | URL | What to download |
|---|---|---|
| Ohio State Extension | ohioline.osu.edu | Turf management, tree care, pest ID guides |
| NALP | landscapeprofessionals.org | Industry best practices |
| ISA (Arborists) | isa-arbor.com/resources | Tree diagnosis guides |
| Penn State Extension | extension.psu.edu | Similar to OSU, very thorough |

### Your Own Data:
- **LMN job templates**: Export service descriptions → embed into ChromaDB
- **Historical job notes**: SingleOps completed job notes → clean + embed
- **Crew SOPs**: Your internal training documents → convert PDF to text → embed

### Converting PDFs to text:
```bash
pip install pypdf2
```
```python
import PyPDF2

def pdf_to_text(filepath: str) -> str:
    reader = PyPDF2.PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Then add to ChromaDB:
doc_text = pdf_to_text("ohio_turfgrass_guide.pdf")
collection.add(ids=["osu-turf-1"], documents=[doc_text])
```

---

## 4. Customer Data → SingleOps CRM

```python
def get_customer(customer_id: str):
    response = requests.get(
        f"https://api.singleops.com/v1/customers/{customer_id}",
        headers={"Authorization": f"Bearer {os.environ['SINGLEOPS_API_KEY']}"}
    )
    return response.json()
```

---

## 5. Tax Rates → Avalara or TaxJar

Ohio's base rate is 7.25% but county/city rates vary.
For accurate invoicing across service areas:

```python
import requests

def get_tax_rate(zip_code: str) -> float:
    # TaxJar API (free tier available)
    response = requests.get(
        f"https://api.taxjar.com/v2/rates/{zip_code}",
        headers={"Authorization": f"Bearer {os.environ['TAXJAR_API_KEY']}"}
    )
    data = response.json()
    return data["rate"]["combined_rate"]
```

---

## 6. Analytics → Snowflake Cortex

Snowflake Cortex is explicitly mentioned in the Granum JD as a
nice-to-have. It's Snowflake's built-in AI/ML layer.

Use case: After 6 months of data, use Cortex to predict:
- Which customers are likely to churn (no booking in 90 days)
- Which services are most profitable by region
- Peak demand periods for crew staffing

```sql
-- Snowflake Cortex example (SQL-based ML)
SELECT
    customer_id,
    SNOWFLAKE.CORTEX.COMPLETE(
        'llama3-70b',
        'Analyze this customer history and predict churn risk: ' || TO_JSON(job_history)
    ) as churn_analysis
FROM customer_job_summary;
```

---

## 7. Feature Flags → LaunchDarkly AI Configs

Also explicitly mentioned in the Granum JD. Allows changing AI
behavior (prompts, model choice, routing logic) without redeploying.

```python
import ldclient
from ldclient.config import Config

ldclient.set_config(Config(os.environ["LAUNCHDARKLY_SDK_KEY"]))
ld = ldclient.get()

# Check which prompt version to use (A/B test)
prompt_variant = ld.variation(
    "orchestrator-prompt-version",
    {"key": "user-123"},
    "v1"  # default
)

if prompt_variant == "v2":
    system_prompt = ORCHESTRATOR_SYSTEM_V2
else:
    system_prompt = ORCHESTRATOR_SYSTEM_V1
```
