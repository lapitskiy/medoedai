import { NextResponse } from "next/server";

const INTERNAL_BACKEND_URL =
  process.env.MEDOED_BACKEND_URL ?? "http://medoedai:5050";

export async function POST(request: Request) {
  try {
    const data = await request.json();

    const flaskRes = await fetch(`${INTERNAL_BACKEND_URL}/api/bot/broadcast`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
      cache: "no-store",
    });

    if (!flaskRes.ok) {
      const errorText = await flaskRes.text();
      return NextResponse.json(
        { success: false, error: `Flask error: ${flaskRes.status} ${errorText}` },
        { status: flaskRes.status },
      );
    }

    const flaskData = await flaskRes.json();
    return NextResponse.json(flaskData);
  } catch (error) {
    console.error("Broadcast POST proxy error:", error);
    return NextResponse.json(
      { success: false, error: String(error) },
      { status: 500 },
    );
  }
}
