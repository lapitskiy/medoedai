import { NextResponse } from "next/server";

const INTERNAL_BACKEND_URL =
  process.env.MEDOED_BACKEND_URL ?? "http://medoedai:5050";

type ProxyRequestOptions = {
  backendPath: string;
  method: "GET" | "POST" | "DELETE";
  request?: Request;
};

export async function proxyBackendRequest({
  backendPath,
  method,
  request,
}: ProxyRequestOptions) {
  try {
    const backendUrl = new URL(`${INTERNAL_BACKEND_URL}${backendPath}`);

    if (request) {
      const incomingUrl = new URL(request.url);
      backendUrl.search = incomingUrl.search;
    }

    const init: RequestInit = {
      method,
      cache: "no-store",
    };

    if (method !== "GET" && request) {
      init.headers = {
        "Content-Type": "application/json",
      };
      init.body = await request.text();
    }

    const response = await fetch(backendUrl, init);
    const body = await response.text();
    const headers = new Headers();

    headers.set(
      "Content-Type",
      response.headers.get("Content-Type") ?? "application/json",
    );

    const contentDisposition = response.headers.get("Content-Disposition");
    if (contentDisposition) {
      headers.set("Content-Disposition", contentDisposition);
    }

    return new NextResponse(body, {
      status: response.status,
      headers,
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
