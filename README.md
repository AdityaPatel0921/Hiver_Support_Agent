# Hiver Support Agent

An AI-based customer support agent built for the Hiver SDE Intern take-home assignment.

The system uses historical customer-support conversations from the Customer Support on Twitter (TWCS) dataset. For this project, I selected `@AppleSupport` and built a retrieval-grounded support workflow that classifies an incoming customer message, retrieves similar historical conversations, generates a response using that context, and decides whether the request should be handled automatically or escalated to a human.

The project is intentionally scoped as a prototype. The focus is on building a practical support workflow, keeping generated responses grounded in historical evidence, making escalation decisions explicit, and evaluating the system instead of relying only on subjective response quality.

---

## 1. Problem Statement

Customer support automation is not just a response-generation problem.

For an incoming customer request, the system should be able to answer three practical questions:

1. What is the customer asking about?
2. How were similar issues handled in historical support conversations?
3. Should this request be handled automatically or reviewed by a human?

This project addresses these questions through:

- intent classification;
- semantic retrieval of historical support conversations;
- retrieval-grounded response generation;
- structured model output;
- auto-handle vs human escalation;
- validation and safety guardrails;
- automated evaluation;
- LLM-based response evaluation.

The goal is not to build a fully autonomous production support system. The goal is to build a small, measurable prototype and understand where automation works and where it should stop.

## 2. Objectives

The system is designed around four main tasks.

### Intent Classification

Classify an incoming customer message into a small set of support intents.

### Historical Retrieval

Retrieve similar customer-support conversations from the selected brand.

### Response Generation

Use the retrieved conversations as evidence when generating a response.

### Support Triage

Decide whether the request should be:

AUTO_HANDLe
ESCALATE

When escalation is selected, the system should provide a reason.


## 3. Project Scope

The project focuses on one customer-support brand from the TWCS dataset:

@AppleSupport

The original TWCS dataset contains approximately 2.8 million tweets across multiple support brands.

Instead of repeatedly processing the complete dataset, I use a manageable AppleSupport subset. This keeps the development and evaluation workflow practical while still working with genuine customer-support conversations.

### Current Working Dataset

 Component | Count 
| AppleSupport support pairs | 5,000 |
| Retrieval / Knowledge Base pairs | 4,850 |
| Held-out evaluation candidates | 150 |
| Embedding dimension | 384 |
| Retrieved examples per query | 3 |

The original raw TWCS CSV is not committed to the repository.

## 4. Why AppleSupport?

I selected @AppleSupport as the support brand because it provides a large number of real customer-support interactions covering different types of customer issues.

This makes it a useful domain for testing:

- intent classification;
- semantic retrieval;
- grounded response generation;
- support triage;
- human escalation.

The objective is not to reproduce Apple's complete support system. The objective is to build and evaluate a focused support-agent workflow using one real support account.

## 5. Dataset

The project uses the **Customer Support on Twitter (TWCS)** dataset.

The relevant fields used during preprocessing include:

tweet_id
author_id
inbound
created_a
text
response_tweet_id
in_response_to_tweet_id

The dataset contains customer messages and responses from customer-support accounts.

For this project, AppleSupport conversations are reconstructed into customer/reply pairs using the conversation relationship fields.

## 6. Data Preparation

The raw dataset is processed in chunks rather than loading the complete CSV into memory at once.

The preprocessing flow is:

TWCS Raw Dataset
       |
       v
Chunked Processing
       |
       v
AppleSupport Filtering
       |
       v
Customer / Reply Matching
       |
       v
Text Sanitization
       |
       v
Validation + Deduplication
       |
       v
5,000 AppleSupport Pairs
       |
       +----------------------+
       |                      |
       v                      v
4,850 Retrieval Pairs     150 Evaluation Candidatesq
