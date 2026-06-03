import { NextResponse } from "next/server";

const INTERNAL_BACKEND_URL =
  process.env.MEDOED_BACKEND_URL ?? "http://medoedai:5050";

export async function DELETE(request: Request) {
  try {
    const body = await request.json();
    const response = await fetch(`${INTERNAL_BACKEND_URL}/api/xgb/prod/models`, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
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
