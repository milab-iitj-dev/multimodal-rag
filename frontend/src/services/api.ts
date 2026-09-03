/**
 * MMRAG Unified — Frontend API Client
 *
 * Communicates exclusively with the unified FastAPI backend via POST /query.
 * No client-side inference, no mock data, no fallback backends.
 *
 * Configuration:
 *   VITE_API_URL  —  Full URL to the /query endpoint
 *                    Default for local dev: http://localhost:8000/query
 */

// ── Request / Response Types ────────────────────────────────

export interface QueryRequest {
  query: string;
  domain: "healthcare" | "scientific" | "auto";
  top_k: number;
  include_images: boolean;
  image_b64?: string;
}

export interface QueryResponse {
  answer: string;
  confidence: number;
  sources: {
    doc_id: string;
    page: number;
    title: string;
    relevance_score: number;
    snippet: string;
  }[];
  retrieval_metadata: {
    method: "fused" | "colpali_only" | "scincl_only";
    scores: { colpali: number; scincl: number; fused: number };
  };
  verification: {
    attribution: boolean;
    faithfulness: boolean;
    confidence_pass: boolean;
  };
  latency_ms: number;
  // Optional fields — rendered only when present in response
  graph?: {
    nodes: { id: string; label: string; type: string }[];
    edges: { source: string; target: string; relation: string }[];
  };
  clinical_note?: string;
  insights?: string[];
}

// ── Fetch with timeout ──────────────────────────────────────

const DEFAULT_TIMEOUT_MS = 60000; // 60 seconds for GPU inference

async function fetchWithTimeout(
  url: string,
  options: RequestInit & { timeout?: number }
): Promise<Response> {
  const { timeout = DEFAULT_TIMEOUT_MS, ...fetchOptions } = options;
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal,
    });
    clearTimeout(id);
    return response;
  } catch (err: any) {
    clearTimeout(id);
    if (err.name === "AbortError") {
      throw new Error(
        `Request timed out after ${timeout / 1000} seconds. ` +
          `The backend may be loading models or processing a heavy query.`
      );
    }
    if (err.message && err.message.includes("Failed to fetch")) {
      throw new Error(
        `Cannot connect to the MMRAG backend at ${url}. ` +
          `Please verify the server is running.`
      );
    }
    throw err;
  }
}

// ── API URL resolution ──────────────────────────────────────

function getApiUrl(): string {
  const url = import.meta.env.VITE_API_URL;
  if (!url) {
    throw new Error(
      "VITE_API_URL is not configured. " +
        "Create a .env.local file with: VITE_API_URL=http://localhost:8000/query"
    );
  }
  return url;
}

// ── Main query function ─────────────────────────────────────

export async function queryPipeline(
  req: QueryRequest
): Promise<QueryResponse> {
  const API_URL = getApiUrl();

  const body: Record<string, unknown> = {
    query: req.query,
    domain: req.domain,
    top_k: req.top_k,
    include_images: req.include_images,
  };

  // Include base64 image only when present
  if (req.image_b64) {
    body.image_b64 = req.image_b64;
  }

  const response = await fetchWithTimeout(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    timeout: DEFAULT_TIMEOUT_MS,
  });

  if (!response.ok) {
    // Try to extract detail from FastAPI error response
    let detail = `${response.status} ${response.statusText}`;
    try {
      const errBody = await response.json();
      if (errBody.detail) {
        detail = errBody.detail;
      }
    } catch {
      // JSON parse failed, use status text
    }
    throw new Error(`Backend error: ${detail}`);
  }

  return (await response.json()) as QueryResponse;
}
