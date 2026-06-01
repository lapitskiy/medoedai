import { proxyBackendRequest } from "@/lib/backend-proxy";

export async function GET(request: Request) {
  return proxyBackendRequest({
    backendPath: "/api/settings/bots/notification-models",
    method: "GET",
    request,
  });
}

export async function POST(request: Request) {
  return proxyBackendRequest({
    backendPath: "/api/settings/bots/notification-models/save",
    method: "POST",
    request,
  });
}
