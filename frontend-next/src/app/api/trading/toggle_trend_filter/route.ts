import { NextResponse } from "next/server";

const INTERNAL_BACKEND_URL = process.env.INTERNAL_BACKEND_URL || "http://192.168.0.21:5050";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const response = await fetch(`${INTERNAL_BACKEND_URL}/api/trading/toggle_trend_filter`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error: any) {
    return NextResponse.json({ success: false, error: error.message }, { status: 500 });
  }
}
