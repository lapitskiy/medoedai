import { proxyBackendRequest } from "@/lib/backend-proxy";

export async function GET(request: Request) {
  return proxyBackendRequest({
    backendPath: "/api/settings/bots/telegram",
    method: "GET",
    request,
  });
}

export async function POST(request: Request) {
  return proxyBackendRequest({
    backendPath: "/api/settings/bots/telegram/save",
    method: "POST",
    request,
  });
}
