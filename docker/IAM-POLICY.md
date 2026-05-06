# IAM Least-Privilege Policy — Tax Authority RAG

## Required Actions

The app container requires exactly three IAM permissions:

| Action | Resource | Purpose |
|---|---|---|
| `bedrock:InvokeModel` | `arn:aws:bedrock:us-east-1::foundation-model/cohere.embed-multilingual-v3` | Generate 1024-dim embeddings for indexing and query |
| `bedrock:InvokeModel` | `arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0` | Cross-encoder reranking of retrieved chunks |
| `bedrock:InvokeModel` | `arn:aws:bedrock:*:780822965578:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0` | Cross-region inference profile for Haiku 4.5 (LLM generator + CRAG grader + NLI judge) |

> **IMPORTANT**: The Haiku 4.5 model ID `us.anthropic.claude-haiku-4-5-20251001-v1:0` uses the
> cross-region inference profile. The resource ARN uses `inference-profile/` not `foundation-model/`.
> Legacy on-demand model IDs (`anthropic.claude-haiku-4-5-*`) are rejected by Bedrock as of 2025.

Additionally for the startup probe:
| `sts:GetCallerIdentity` | `*` | Validate credentials at container startup; logs ARN |

## IAM Policy JSON

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockCohereEmbed",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "arn:aws:bedrock:us-east-1::foundation-model/cohere.embed-multilingual-v3"
    },
    {
      "Sid": "BedrockCohereRerank",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"
    },
    {
      "Sid": "BedrockHaikuCrossRegionProfile",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "arn:aws:bedrock:*:780822965578:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    },
    {
      "Sid": "STSStartupProbe",
      "Effect": "Allow",
      "Action": "sts:GetCallerIdentity",
      "Resource": "*"
    }
  ]
}
```

## Deployment Notes

1. Create an IAM user `tax-rag-eval` with programmatic access only (no console login).
2. Attach the above inline policy.
3. Generate access keys and populate `.env` (gitignored).
4. For CI/CD, use GitHub Actions OIDC + `sts:AssumeRoleWithWebIdentity` instead of static keys.
5. For production, attach the policy to an ECS task role and use IAM instance profiles — no static keys.

## Current Assignment Context

Currently using root credentials for account `780822965578` — flagged in MASTER-PLAN §A.
This is acceptable for the assignment only. Before any team-shared work:
1. Create the `tax-rag-eval` IAM user per above.
2. Delete or disable root access keys.
3. Enable MFA on the root account.
