# API Contracts

All endpoints are prefixed with `/api/`. Authentication uses JWT Bearer tokens unless noted as public.

---

## Authentication

### `POST /api/token/`

Obtain a JWT access + refresh token pair.

**Auth:** Public

**Request body:**
```json
{
  "email": "user@example.com",
  "password": "string"
}
```

**Response `200`:**
```json
{
  "access": "<jwt_access_token>",
  "refresh": "<jwt_refresh_token>"
}
```

**Errors:** `401` — invalid credentials

---

### `POST /api/token/refresh/`

Exchange a valid refresh token for a new access token.

**Auth:** Public

**Request body:**
```json
{
  "refresh": "<jwt_refresh_token>"
}
```

**Response `200`:**
```json
{
  "access": "<new_jwt_access_token>"
}
```

**Errors:** `401` — refresh token invalid or expired

---

## Accounts

### `POST /api/accounts/register/`

Create a new user account.

**Auth:** Public

**Request body:**
```json
{
  "email": "user@example.com",
  "password": "string (min 8 chars)"
}
```

**Response `201`:**
```json
{
  "id": "<uuid>",
  "email": "user@example.com"
}
```

**Errors:** `400` — validation error (duplicate email, weak password)

---

### `GET /api/accounts/me/`

Retrieve the authenticated user's profile.

**Auth:** Bearer token required

**Response `200`:**
```json
{
  "id": "<uuid>",
  "email": "user@example.com",
  "first_name": "string",
  "last_name": "string",
  "date_joined": "2026-01-01T00:00:00Z"
}
```

**Errors:** `401` — missing or invalid token

---

### `PATCH /api/accounts/me/`

Update the authenticated user's profile fields.

**Auth:** Bearer token required

**Request body** (all fields optional):
```json
{
  "first_name": "string",
  "last_name": "string",
  "email": "user@example.com"
}
```

**Response `200`:** Updated user object (same shape as `GET /api/accounts/me/`)

**Errors:** `400` — validation error, `401` — unauthorised

---

## Model Registry

The registry stores the browser-runnable model catalog and records
inference-run metadata reported by the client. **The backend never runs
inference** — GPU compute happens in the browser via WebGPU (see
`docs/explanations/webgpu-inference.md`).

### `GET /api/registry/models/`

List model cards. Anonymous callers see only public models; authenticated
callers also see their own private models.

**Auth:** Public (read); Bearer token required to see private models

**Response `200`:**
```json
[
  {
    "id": "<uuid>",
    "slug": "tiny-matmul",
    "name": "Tiny MatMul",
    "task": "custom",
    "description": "A demo dense layer kernel.",
    "weights_url": "https://cdn.example.com/tiny.safetensors",
    "config": { "entry_point": "main", "workgroup_size": [16, 16, 1] },
    "size_bytes": 1048576,
    "license": "MIT",
    "is_public": true,
    "created_at": "2026-07-02T00:00:00Z",
    "updated_at": "2026-07-02T00:00:00Z"
  }
]
```

`task` is one of `llm | vision | embedding | audio | custom`. `config` is
free-form JSON the browser runtime consumes (tensor shapes, quantization, WGSL
entry points, tokenizer refs, preprocessing params).

---

### `POST /api/registry/models/`

Create a model card (owned by the caller).

**Auth:** Bearer token required

**Request body:**
```json
{
  "slug": "my-net",
  "name": "My Net",
  "task": "vision",
  "description": "string",
  "weights_url": "https://…",
  "config": {},
  "size_bytes": null,
  "license": "Apache-2.0",
  "is_public": true
}
```

**Response `201`:** The created model card.

**Errors:** `400` — validation error (e.g. duplicate `slug`), `401` — unauthorised

---

### `GET /api/registry/models/{slug}/`

Retrieve one model card by slug.

**Auth:** Public for public models; Bearer token required for private ones

**Response `200`:** A model card object. **Errors:** `404` — not found / not visible

`PUT` / `PATCH` / `DELETE` on the same URL update or remove a card the caller owns
(Bearer token required).

---

### `POST /api/registry/runs/`

Record the outcome of an in-browser inference run.

**Auth:** Bearer token required

**Request body:**
```json
{
  "model": "<model_uuid>",
  "status": "completed",
  "params": { "prompt": "hello", "max_tokens": 64 },
  "metrics": { "latency_ms": 12.5, "tokens_per_sec": 48.0, "gflops": 340.0 }
}
```

`status` is `completed | failed` (defaults to `completed`).

**Response `201`:**
```json
{
  "id": "<uuid>",
  "model": "<model_uuid>",
  "model_slug": "tiny-matmul",
  "status": "completed",
  "params": { "prompt": "hello", "max_tokens": 64 },
  "metrics": { "latency_ms": 12.5, "tokens_per_sec": 48.0, "gflops": 340.0 },
  "created_at": "2026-07-02T00:00:00Z"
}
```

**Errors:** `400` — validation error, `401` — unauthorised

---

### `GET /api/registry/runs/`

List the authenticated user's own inference runs (most recent first).

**Auth:** Bearer token required

**Response `200`:** Array of run objects (same shape as the `POST` response).

---

## Health

### `GET /api/health/`

Service liveness check.

**Auth:** Public

**Response `200`:**
```json
{
  "status": "ok"
}
```

---

## Common HTTP Status Codes

| Code | Meaning |
|------|---------|
| `200` | OK — request succeeded |
| `201` | Created — resource created |
| `400` | Bad Request — validation error |
| `401` | Unauthorised — missing or invalid JWT |
| `403` | Forbidden — authenticated but insufficient permissions |
| `404` | Not Found |
| `500` | Internal Server Error |

---

## Request Headers

All authenticated requests must include:

```
Authorization: Bearer <access_token>
Content-Type: application/json
```

---

## Model Notes

- All primary keys are UUIDs (`uuid4`)
- Timestamps are ISO 8601 in UTC
- `CustomUser` extends Django's `AbstractUser` — `email` is the login identifier (no `username` field)
