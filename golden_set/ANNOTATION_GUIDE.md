# Golden Evaluation Set: Human Annotation Guide & Taxonomy Reference

**Target Brand:** `@AppleSupport`  
**Dataset Reference:** Twitter Customer Support (`twcs.csv`)  
**Evaluation Scope:** 150 Held-Out Diverse Support Inquiries  
**Policy Standard:** Genuine Manual Human Review (Zero Automated Label Fabrication)  

---

## 1. Purpose of the Golden Set

The Golden Evaluation Set serves as the **uncontaminated, ground-truth benchmark** for evaluating the `@AppleSupport` AI customer support agent. It measures the system's real-world ability to:
1. Accurately categorize unconstrained customer intents without keyword overfitting.
2. Make safety-critical triage decisions (`AUTO_HANDLE` vs `ESCALATE`) prioritizing customer hardware and security protection.
3. Measure inter-rater agreement (Cohen's $\kappa$) between human decisions and model predictions.
4. Establish unbiased grounded reply quality metrics completely free from retrieval test-set leakage.

---

## 2. Sampling Methodology

The 150 candidate evaluation examples were sampled using the following protocol:
- **Held-Out Partitioning:** Separated deterministically (`random_state=42`) from the raw/processed dataset *before* the FAISS vector index was constructed.
- **Stratified Coverage:** Balanced across the 5 target intent domains (~30 samples per domain) to prevent majority-class dominance.
- **Deduplication:** Sanitized to eliminate near-duplicate phrasings and identical customer handles.
- **Traceability:** Every candidate preserves its original `customer_tweet_id`, raw `customer_text`, and historical `apple_reply`.

---

## 3. Why 150 Examples Were Selected

- **Statistical Significance:** A sample size of $N=150$ provides sufficient statistical power to compute Macro-F1 across 5 classes with meaningful per-class support ($N=30$ per class) and tightly bounded confidence intervals.
- **Human Feasibility:** 150 examples represents a rigorous, realistic volume for thorough, high-attention manual human annotation without annotator fatigue.
- **Assignment Compliance:** Directly satisfies the Hiver assignment requirement of a *"150–250 hand-labelled examples you built yourself"*.

---

## 4. The 5 Intent Definitions & Taxonomy

| Intent Class | Definition | Inclusion Criteria | Exclusion Criteria | Canonical Example |
| :--- | :--- | :--- | :--- | :--- |
| **`OS_SOFTWARE_UPDATE`** | Issues stemming from iOS, iPadOS, macOS, or watchOS installations, updates, or software bugs. | Update verification loops, OS feature lag, beta profile issues, update installation errors, Wi-Fi drops post-update. | Physical glass damage occurring during an update; Apple ID password lockout. | *"My iPhone won't update to iOS 17.1. It gets stuck on 'Verifying Update' for hours."* |
| **`BATTERY_POWER`** | Inquiries regarding battery health, capacity degradation, standard charging behavior, and power drain. | Rapid battery drain, unexpected shutdown at 20%, maximum capacity drop under 80%, battery settings walkthroughs. | Battery swelling, bulging chassis, smoke/spark hazard (classified under `HARDWARE_DAMAGE`). | *"My iPhone 14 Pro battery health dropped from 99% to 85% in just 2 months!"* |
| **`ACCOUNT_ICLOUD`** | Digital identity, authentication, security credentials, and cloud storage subscriptions. | Apple ID password resets, two-factor authentication (2FA), iCloud backup storage full, Family Sharing access. | Physical device theft without digital account compromise. | *"I forgot my Apple ID password and the recovery phone number is no longer active."* |
| **`HARDWARE_DAMAGE`** | Physical breakage, thermal runaway hazards, liquid ingress, and mechanical failures. | Shattered/cracked screen, water damage, broken keyboard keys, battery swelling, burning smell, smoke. | Software freeze where screen is responsive after a hard restart. | *"My iPhone battery is swelling and pushing the screen out from the frame!"* |
| **`GENERAL_INQUIRY`** | Informational requests, compatibility questions, sales/trade-in inquiries, and store logistics. | Trade-in value estimates, USB-C display specs, Apple Care terms, retail store hours, engraving options. | Specific diagnostic troubleshooting of malfunctioning hardware/software. | *"Does the new iPhone 15 support USB-C display output to external 4K monitors?"* |

---

## 5. Triage Action Rules: `AUTO_HANDLE` vs `ESCALATE`

### A. When to Auto-Handle (`AUTO_HANDLE`)
Auto-handle should be selected when an automated agent can safely guide the customer through verified, self-service Apple documentation:
- Routine OS settings navigation (e.g., *Settings > General > Software Update*).
- Standard battery health checks (e.g., *Settings > Battery > Battery Health*).
- iCloud storage breakdown and backup settings.
- Direct answers to device specifications and trade-in URLs.
- **Requirement:** `ground_truth_reason` must be empty (`""` or `None`).

### B. Mandatory Escalation Triggers (`ESCALATE`)
Human escalation is required whenever automated handling poses physical danger, security risk, legal liability, or severe brand damage:
1. **Critical Safety & Thermal Hazards:** Battery swelling, bulging chassis, excessive burning heat, smoke, or spark risk.
2. **Physical Hardware Breakage:** Cracked screens, liquid immersion, broken buttons requiring physical Genius Bar / Mail-in inspection.
3. **Account Security Compromise:** Unauthorized account access, hacked Apple ID, unrecognized 2FA prompts, fraudulent charges.
4. **Legal / Regulatory / Severe Complaints:** Mentions of lawsuits, consumer protection complaints, or extreme distress.
5. **Requirement:** `ground_truth_reason` is **mandatory** and must explicitly state the escalation trigger.

---

## 6. Urgency Scale (1–5) Guidelines

| Score | Urgency Level | Description | Criteria & Examples |
| :---: | :--- | :--- | :--- |
| **1** | Informational / General | Routine informational questions with zero operational impact. | General specs, trade-in estimates, retail store hours. |
| **2** | Low / Standard Support | Common software or battery questions with active device usability. | Update verification questions, standard battery optimization. |
| **3** | Moderate / Non-Urgent Issue | Device features partially degraded or routine account configuration. | iCloud backup space warnings, Family Sharing permission sync. |
| **4** | High / Broken Hardware or Security | Physical device impairment or potential credential security risks. | Shattered display glass, lost Apple ID password, suspected account access. |
| **5** | Critical / Emergency Hazard | Active physical safety danger, thermal runaway, or severe compromise. | Battery swelling, smoking charger, active financial fraud. |

---

## 7. Escalation Reason Requirements

- When `ground_truth_action == "ESCALATE"`, the human annotator **must provide a concise, factual reason** in `ground_truth_reason`.
- **Good Examples:**
  - *"Battery swelling is an active fire/explosion hazard requiring immediate device isolation and specialist handling."*
  - *"Shattered OLED display requires physical hardware inspection and Authorized Service repair."*
  - *"Customer reports unauthorized iCloud login attempt indicating potential credential compromise."*
- **Bad Examples (Avoid):**
  - *"Needs help"* (Too vague)
  - *"Escalate"* (Restates action)

---

## 8. Handling Ambiguous & Compound Cases

1. **Safety Overrides Software:** If a customer mentions an OS update but also reports physical screen breakage (*"Updated to iOS 17 and dropped my phone, screen is shattered"*), classify as `HARDWARE_DAMAGE` and `ESCALATE`.
2. **Thermal Severity Threshold:**
   - *"My phone gets warm while gaming"* $\rightarrow$ `BATTERY_POWER` / `AUTO_HANDLE` / Urgency 2.
   - *"My phone gets scorching hot and smells like burning plastic"* $\rightarrow$ `HARDWARE_DAMAGE` / `ESCALATE` / Urgency 5.
3. **Sarcasm / Angry Tone:** If a customer is expressing frustration about a routine software bug without safety hazards, classify as `OS_SOFTWARE_UPDATE` / `AUTO_HANDLE`, but note customer sentiment in `annotator_notes`.

---

## 9. Annotator Notes Usage

The `annotator_notes` field is reserved for the human annotator to document:
- Edge-case justifications.
- Secondary intent co-occurrences.
- Specific phrasing ambiguities.

---

## 10. Mandatory Human Annotation Statement

> [!IMPORTANT]
> **Human Integrity Verification Statement:**  
> All records in `golden_set/golden_eval_set.csv` must be reviewed and assigned by a human engineer. Automated heuristic outputs (`suggested_*`) are provided solely as an optional reading aid in the template and **must never be treated as ground truth** without deliberate human verification.
