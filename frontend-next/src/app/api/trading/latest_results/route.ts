import { NextResponse } from "next/server";

const INTERNAL_BACKEND_URL =
  process.env.MEDOED_BACKEND_URL ?? "http://medoedai:5050";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const params = new URLSearchParams();

    const symbol = searchParams.get("symbol");
    if (symbol) {
      params.set("symbol", symbol);
    }

    const query = params.toString();
    const response = await fetch(
      `${INTERNAL_BACKEND_URL}/api/trading/latest_results${query ? `?${query}` : ""}`,
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
