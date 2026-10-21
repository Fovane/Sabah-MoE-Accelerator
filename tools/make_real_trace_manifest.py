import json
import os

items = [
    {
        "id": "py-window",
        "n_predict": 64,
        "text": (
            "You are helping debug a production Python service. Here is the code and the traceback.\n\n"
            "```python\n"
            "def rolling_max(xs, k):\n"
            "    out = []\n"
            "    for i in range(len(xs) - k):\n"
            "        out.append(max(xs[i:i + k]))\n"
            "    return out\n\n"
            "def p95(values):\n"
            "    s = sorted(values)\n"
            "    return s[int(0.95 * len(s))]\n\n"
            "series = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]\n"
            "print(rolling_max(series, 3))\n"
            "print(p95(rolling_max(series, 3)))\n"
            "```\n\n"
            "Traceback (most recent call last):\n"
            "  File \"stats.py\", line 12, in <module>\n"
            "    print(p95(rolling_max(series, 3)))\n"
            "  File \"stats.py\", line 7, in p95\n"
            "    return s[int(0.95 * len(s))]\n"
            "IndexError: list index out of range\n\n"
            "Unit tests also report that the last window is missing from rolling_max "
            "and that p95 disagrees with numpy.percentile on small samples. "
            "Find the bugs, explain why each one is wrong, and give a corrected "
            "implementation that keeps the public signatures and handles empty input."
        ),
    },
    {
        "id": "cuda-reduce",
        "n_predict": 64,
        "text": (
            "Review this CUDA kernel used in a temperature simulation. It is correct "
            "for even block sizes but returns wrong sums for odd ones, and the "
            "profiler shows a large tail effect.\n\n"
            "```cuda\n"
            "__global__ void block_sum(const float * in, float * out, int n) {\n"
            "    extern __shared__ float s[];\n"
            "    int t = threadIdx.x;\n"
            "    s[t] = in[blockIdx.x * blockDim.x + t];\n"
            "    __syncthreads();\n"
            "    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {\n"
            "        if (t < stride) s[t] += s[t + stride];\n"
            "        __syncthreads();\n"
            "    }\n"
            "    if (t == 0) out[blockIdx.x] = s[0];\n"
            "}\n\n"
            "block_sum<<<grid, 256, 256 * sizeof(float)>>>(d_in, d_out, n);\n"
            "```\n\n"
            "The host code launches grid = (n + 255) / 256 without guarding the last "
            "block, and the call sites pass n values that are not multiples of 256. "
            "Explain the out-of-bounds read, the odd-size failure, and the tail "
            "effect, then rewrite the kernel with a bounds check and a warp-shuffle "
            "reduction. State the occupancy and bank-conflict implications of your "
            "version."
        ),
    },
    {
        "id": "sql-slow",
        "n_predict": 64,
        "text": (
            "A PostgreSQL 16 analytics query went from 1.2 s to 47 s after a data "
            "migration. The tables are orders(id, customer_id, created_at, total), "
            "order_items(order_id, sku, qty, unit_price) and customers(id, region, "
            "segment). orders has 40 million rows, order_items 210 million, "
            "customers 900 thousand. All foreign keys have indexes, orders.created_at "
            "has a btree index, and the migration changed order_items.unit_price from "
            "numeric(12,2) to double precision.\n\n"
            "```sql\n"
            "SELECT c.region, date_trunc('month', o.created_at) AS m,\n"
            "       sum(i.qty * i.unit_price) AS revenue,\n"
            "       count(DISTINCT o.customer_id) AS buyers\n"
            "FROM orders o\n"
            "JOIN order_items i ON i.order_id = o.id\n"
            "JOIN customers c ON c.id = o.customer_id\n"
            "WHERE o.created_at >= now() - interval '18 months'\n"
            "  AND c.segment = 'enterprise'\n"
            "GROUP BY 1, 2\n"
            "ORDER BY 2, 1;\n"
            "```\n\n"
            "EXPLAIN shows a nested loop over 40 million rows with a filter on "
            "customers applied after the join. Rewrite the query and say which "
            "statistics or indexes you would add, and why the type change could "
            "affect the plan."
        ),
    },
    {
        "id": "ps-deploy",
        "n_predict": 64,
        "text": (
            "This PowerShell 7 deployment script works on the build server but fails "
            "on the release server with \"Cannot bind argument to parameter 'Path' "
            "because it is null\" and then leaves a locked file behind.\n\n"
            "```powershell\n"
            "param([string]$Root = $env:DEPLOY_ROOT, [string]$Version)\n"
            "Get-ChildItem -Path $Root -Recurse -Filter *.dll | ForEach-Object {\n"
            "    $target = Join-Path $ReleaseDir $_.Name\n"
            "    Copy-Item $_.FullName $target -Force\n"
            "}\n"
            "$zip = Join-Path $Root \"artifacts\\app-$Version.zip\"\n"
            "Compress-Archive -Path (Join-Path $ReleaseDir '*') -DestinationPath $zip -Force\n"
            "Invoke-WebRequest -Uri $env:UPLOAD_URL -Method Put -InFile $zip\n"
            "Remove-Item $Root -Recurse -Force\n"
            "```\n\n"
            "The environment variables are set at machine scope, the script runs as "
            "a scheduled task under a service account, and $ReleaseDir is never "
            "assigned. Find every bug, including the failure mode on an empty "
            "directory, and give a corrected script with error handling and "
            "idempotent retries."
        ),
    },
    {
        "id": "cpp-template",
        "n_predict": 64,
        "text": (
            "Explain this C++20 compile error and give the smallest correct fix.\n\n"
            "```cpp\n"
            "#include <concepts>\n"
            "#include <vector>\n"
            "#include <ranges>\n\n"
            "template <std::integral T>\n"
            "struct histogram {\n"
            "    std::vector<T> bins;\n"
            "    explicit histogram(std::size_t n) : bins(n, T{}) {}\n"
            "    void add(T v) { ++bins[static_cast<std::size_t>(v)]; }\n"
            "};\n\n"
            "template <std::ranges::input_range R>\n"
            "auto counts(R&& r, std::size_t n) {\n"
            "    histogram h(n);\n"
            "    for (auto&& v : r) h.add(v);\n"
            "    return h;\n"
            "}\n"
            "```\n\n"
            "gcc 14 reports \"class template argument deduction failed\" at the "
            "histogram declaration, and clang 18 additionally reports that the "
            "constraint on add is not satisfied when the range yields a reference. "
            "The intent is that counts works for any integral input range and for "
            "standard containers. Explain the deduction failure, the reference "
            "problem, and the fix, and note whether a deduction guide or an "
            "explicit template argument is the better design."
        ),
    },
]

out = r"results\phase0\real-trace\manifest.jsonl"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")
print(out, len(items), "items")
