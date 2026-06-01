import { proxyBackendRequest } from "@/lib/backend-proxy";

export async function GET(request: Request) {
  return proxyBackendRequest({
    backendPath: "/api/settings/bots/max",
    method: "GET",
    request,
  });
}

export async function POST(request: Request) {
  return proxyBackendRequest({
    backendPath: "/api/settings/bots/max/save",
    method: "POST",
    request,
  });
}
