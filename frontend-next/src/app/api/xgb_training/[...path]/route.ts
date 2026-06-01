import { NextResponse } from "next/server";

import { proxyBackendRequest } from "@/lib/backend-proxy";

type RouteParams = {
  path: string[];
};

const getRoutes = new Map<string, string>([
  ["symbols", "/api/xgb/symbols"],
  ["runs", "/api/xgb/runs"],
  ["grid_full_preset", "/api/xgb/grid_full_preset"],
  ["active_training", "/training/xgb_active_training"],
  ["grid_status", "/training/xgb_grid_status"],
]);

const postRoutes = new Map<string, string>([
  ["grid_full_preset", "/api/xgb/grid_full_preset"],
  ["train_grid_task", "/training/train_xgb_grid_task"],
  ["train_grid_full", "/training/train_xgb_grid_full"],
  ["create_model_version", "/create_xgb_model_version"],
]);

function notFoundResponse() {
  return NextResponse.json(
    {
      success: false,
      error: "Route is not supported",
    },
    { status: 404 },
  );
}

export async function GET(
  request: Request,
  { params }: { params: Promise<RouteParams> },
) {
  const { path } = await params;
  const joinedPath = path.join("/");
  const backendPath = getRoutes.get(joinedPath);

  if (!backendPath) {
    return notFoundResponse();
  }

  return proxyBackendRequest({
    backendPath,
    method: "GET",
    request,
  });
}

export async function POST(
  request: Request,
  { params }: { params: Promise<RouteParams> },
) {
  const { path } = await params;
  const joinedPath = path.join("/");
  const backendPath = postRoutes.get(joinedPath);

  if (!backendPath) {
    return notFoundResponse();
  }

  return proxyBackendRequest({
    backendPath,
    method: "POST",
    request,
  });
}
