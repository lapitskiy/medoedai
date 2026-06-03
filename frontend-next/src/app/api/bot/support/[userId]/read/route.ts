import { NextRequest, NextResponse } from "next/server";

const INTERNAL_BACKEND_URL = process.env.MEDOED_BACKEND_URL ?? "http://medoedai:5050";

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ userId: string }> }
) {
  try {
    const { userId } = await context.params;
    const res = await fetch(`${INTERNAL_BACKEND_URL}/api/bot/support/${userId}/read`, {
      method: "POST",
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (error) {
    return NextResponse.json({ success: false, error: String(error) }, { status: 500 });
  }
}
