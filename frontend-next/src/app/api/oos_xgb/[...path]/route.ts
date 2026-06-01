import { NextResponse } from "next/server";

import { proxyBackendRequest } from "@/lib/backend-proxy";

type RouteParams = {
  path: string[];
};

const getRoutes = new Map<string, string>([
  ["runs", "/api/xgb/oos/runs"],
  ["wf_runs", "/api/xgb/wf/runs"],
  ["test_status", "/xgb_oos_test_status"],
  ["batch_status", "/xgb_oos_batch_status"],
  ["batch_results", "/xgb_oos_batch_results"],
  ["recent_batches", "/xgb_oos_recent_batches"],
  ["csv_list", "/xgb_oos_csv_list"],
]);

const postRoutes = new Map<string, string>([
  ["test_async", "/xgb_oos_test_async"],
  ["delete_runs", "/xgb_oos_delete_runs"],
  ["copy_to_prod", "/xgb_oos_copy_to_prod"],
  ["copy_to_wf", "/xgb_oos_copy_to_wf"],
  ["prune_runs", "/xgb_oos_prune_runs"],
  ["prune_duplicates", "/xgb_oos_prune_duplicates"],
  ["batch_async", "/xgb_oos_batch_async"],
  ["batch_save_csv", "/xgb_oos_batch_save_csv"],
  ["batch_force_finalize", "/xgb_oos_batch_force_finalize"],
  ["batch_cancel", "/xgb_oos_batch_cancel"],
  ["save_experiment", "/xgb_oos_save_experiment"],
  ["save_top_csv_experiment", "/xgb_oos_save_top_csv_experiment"],
  ["csv_delete", "/xgb_oos_csv_delete"],
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

  if (joinedPath.startsWith("csv_download/")) {
    const filename = path.slice(1).join("/");
    if (!filename) {
      return notFoundResponse();
    }

    return proxyBackendRequest({
      backendPath: `/xgb_oos_csv_download/${encodeURIComponent(filename)}`,
      method: "GET",
      request,
    });
  }

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
