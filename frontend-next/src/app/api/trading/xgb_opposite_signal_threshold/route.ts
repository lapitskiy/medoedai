import { NextResponse } from "next/server";

const INTERNAL_BACKEND_URL =
  process.env.MEDOED_BACKEND_URL ?? "http://medoedai:5050";

export async function POST(request: Request) {
  try {
    const body = await request.text();
    const response = await fetch(`${INTERNAL_BACKEND_URL}/api/trading/xgb_opposite_signal_threshold`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body,
      cache: "no-store",
    });

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
