import { NextResponse } from "next/server";

const INTERNAL_BACKEND_URL =
  process.env.MEDOED_BACKEND_URL ?? "http://medoedai:5050";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = searchParams.get("limit") ?? "200";
  const symbol = searchParams.get("symbol") ?? "";
  const sessionId = searchParams.get("session_id") ?? "";
  const modelPath = searchParams.get("model_path") ?? "";

  const params = new URLSearchParams({ limit });
  if (symbol) {
    params.set("symbol", symbol);
  }
  if (sessionId) {
    params.set("session_id", sessionId);
  }
  if (modelPath) {
    params.set("model_path", modelPath);
  }

  try {
    const response = await fetch(
      `${INTERNAL_BACKEND_URL}/api/xgb/predictions?${params.toString()}`,
      {
        cache: "no-store",
      },
    );

    const text = await response.text();

    return new NextResponse(text, {
      status: response.status,
      headers: {
        "Content-Type": response.headers.get("Content-Type") ?? "application/json",
      },
    });
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Proxy request failed",
      },
      { status: 502 },
    );
  }
}
