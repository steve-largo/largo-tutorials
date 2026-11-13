"""Self-contained 48-chunk infrastructure corpus and 28 two-slice evaluation queries."""

from __future__ import annotations

from schemas import CorpusChunk, EvalQuery

CORPUS_CHUNKS: tuple[CorpusChunk, ...] = (
    # --- Domain 1: tls-ingress (6 chunks) ---
    CorpusChunk(
        chunk_id="tls-alpn-record-mismatch",
        domain="tls-ingress",
        title="Diagnosing ERR_SSL_PROTOCOL_ERROR on Upstream Listeners",
        text=(
            "Clients surface ERR_SSL_PROTOCOL_ERROR on port 443 when an ingress listener emits "
            "plaintext HTTP bytes or a malformed TLS record header prior to ServerHello negotiation. "
            "Verify that nginx http2 or ssl directives are enabled on the target socket and confirm "
            "PROXY protocol headers are not forwarded to a backend expecting raw TLS frames."
        ),
    ),
    CorpusChunk(
        chunk_id="tls-x509-missing-intermediate",
        domain="tls-ingress",
        title="Resolving X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT_LOCALLY",
        text=(
            "Automated Go, Python, and curl clients fail with "
            "X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT_LOCALLY when the server presents only its leaf "
            "certificate without concatenating intermediate CA certificates into fullchain.pem. "
            "Desktop browsers often mask this misconfiguration via Authority Information Access "
            "caching, whereas non-browser TLS stacks reject the incomplete chain immediately."
        ),
    ),
    CorpusChunk(
        chunk_id="tls-ocsp-stapling-timeout",
        domain="tls-ingress",
        title="Issuer-Signed Revocation Stapling on Edge Proxies",
        text=(
            "Edge proxies attach cached cryptographic revocation assertions signed by certificate "
            "issuers directly inside TLS handshake extensions. Background responder refresh loops "
            "renew time-bounded status proofs asynchronously so mutual verification enforces "
            "credential validity without synchronous third-party directory lookups during session setup."
        ),
    ),
    CorpusChunk(
        chunk_id="tls-mtls-cert-rotation",
        domain="tls-ingress",
        title="Zero-Downtime mTLS Trust Bundle Rotation",
        text=(
            "During root CA rollover across service mesh workloads, distribute a dual-root trust "
            "bundle containing both old and incoming CA certificates before issuing leaf credentials "
            "signed by the new authority. Removing the retiring CA prior to full fleet propagation "
            "triggers peer verification failures across active mutual TLS channels."
        ),
    ),
    CorpusChunk(
        chunk_id="tls-session-ticket-keys",
        domain="tls-ingress",
        title="TLS 1.3 Session Ticket Key Rotation Across Load Balancers",
        text=(
            "Stateless TLS 1.3 resumption relies on symmetric session ticket encryption keys shared "
            "across all ingress replicas. If pods generate independent ephemeral ticket keys at boot, "
            "abbreviated handshakes routed to peer replicas fall back to full asymmetric key exchange "
            "and inflate p99 handshake CPU utilization."
        ),
    ),
    CorpusChunk(
        chunk_id="tls-sni-routing-mismatch",
        domain="tls-ingress",
        title="SNI Hostname Routing and Default Certificate Fallback",
        text=(
            "Multi-tenant TLS gateways inspect the plaintext Server Name Indication extension during "
            "ClientHello to select the matching virtual host certificate. Load probes that connect "
            "by raw IP address omit SNI and receive the gateway fallback certificate, causing "
            "SAN hostname verification errors."
        ),
    ),
    # --- Domain 2: gpu-compute (6 chunks) ---
    CorpusChunk(
        chunk_id="gpu-sku-g5-12xlarge",
        domain="gpu-compute",
        title="AWS g5.12xlarge Instance PCIe Topology and Interconnect Limits",
        text=(
            "The g5.12xlarge instance pairs four NVIDIA A10G 24 GB GPUs (96 GB aggregate VRAM) with "
            "48 vCPUs and 192 GiB host RAM. Because A10G boards on g5.12xlarge lack NVLink bridges, "
            "all tensor-parallel all-reduce collectives traverse PCIe Gen4 lanes, making pipeline "
            "parallelism or 4-way quantized sharding preferable over high-frequency ring reductions."
        ),
    ),
    CorpusChunk(
        chunk_id="gpu-sku-g5-2xlarge",
        domain="gpu-compute",
        title="Single-GPU g5.2xlarge Sizing for 7B-8B Inference",
        text=(
            "An AWS g5.2xlarge node provides one NVIDIA A10G GPU with 24 GB GDDR6 memory, 600 GB/s "
            "memory bandwidth, and 8 vCPUs. It fits an 8-billion-parameter FP8 or AWQ INT4 model "
            "comfortably with 8 GB to 14 GB remaining for paged KV cache blocks."
        ),
    ),
    CorpusChunk(
        chunk_id="gpu-sku-p4d-24xlarge",
        domain="gpu-compute",
        title="AWS p4d.24xlarge NVSwitch and EFA Multi-Node Architecture",
        text=(
            "Each p4d.24xlarge instance houses eight NVIDIA A100 40 GB GPUs linked via 600 GB/s "
            "bidirectional NVSwitch fabric alongside four 100 Gbps Elastic Fabric Adapter NICs. "
            "Use p4d.24xlarge for distributed pretraining or 8-way tensor parallelism where PCIe "
            "interconnects become communication-bound."
        ),
    ),
    CorpusChunk(
        chunk_id="nccl-ib-disable-env",
        domain="gpu-compute",
        title="Fallback Transport Configuration via NCCL_IB_DISABLE=1",
        text=(
            "Setting NCCL_IB_DISABLE=1 forces NVIDIA Collective Communications Library workers to "
            "bypass InfiniBand and RoCE verbs interfaces, routing inter-node collectives over standard "
            "POSIX TCP sockets via NCCL_SOCKET_IFNAME. Use this flag when unprivileged containers "
            "lack RDMA uverbs device mounts or encounter mlx5 completion queue timeouts."
        ),
    ),
    CorpusChunk(
        chunk_id="nccl-p2p-nvl-ring",
        domain="gpu-compute",
        title="Intra-Node Peer-to-Peer BAR1 and NVSwitch Collective Fabrics",
        text=(
            "Direct peer-to-peer DMA apertures transfer tensors across multi-GPU accelerators on "
            "the same host via NVSwitch or PCIe switches without staging buffers through system "
            "DRAM or root complexes. Enabling NCCL_P2P_LEVEL=NVL sustains sub-microsecond barrier "
            "synchronization during intra-box all-gather and reduce-scatter phases."
        ),
    ),
    CorpusChunk(
        chunk_id="cuda-launch-blocking-debug",
        domain="gpu-compute",
        title="Pinpointing Asynchronous Device Faults with CUDA_LAUNCH_BLOCKING=1",
        text=(
            "Because CUDA kernel launches are asynchronous with respect to Python host threads, "
            "illegal memory accesses surface dozens of ops later at the next stream synchronization. "
            "Exporting CUDA_LAUNCH_BLOCKING=1 serializes every kernel dispatch on the host CPU so "
            "stack traces point to the exact offending operator, at the cost of throughput."
        ),
    ),
    # --- Domain 3: llm-serving (6 chunks) ---
    CorpusChunk(
        chunk_id="vllm-flag-batched-tokens",
        domain="llm-serving",
        title="Tuning --max-num-batched-tokens for vLLM Iteration Budgets",
        text=(
            "The --max-num-batched-tokens CLI flag caps the sum of prefill and decode tokens "
            "scheduled into a single vLLM forward pass. Raising --max-num-batched-tokens above 8192 "
            "boosts offline ingestion throughput on H100 GPUs, whereas lowering it toward 2048 bounds "
            "per-step compute duration and protects inter-token latency."
        ),
    ),
    CorpusChunk(
        chunk_id="vllm-chunked-prefill-sched",
        domain="llm-serving",
        title="Iteration-Level Chunked Prefill Scheduling",
        text=(
            "Iteration-level continuous schedulers partition multi-thousand-token context encodings "
            "into bounded fixed-size slices and interleave those slices alongside single-token "
            "autoregressive decode steps. Disaggregating heavy context ingestion prevents compute-bound "
            "attention passes from monopolizing SMs and spiking inter-token tail latency."
        ),
    ),
    CorpusChunk(
        chunk_id="vllm-prefix-kv-caching",
        domain="llm-serving",
        title="Radix-Hashed Paged KV Block Deduplication",
        text=(
            "Paged attention engines hash fixed-length token blocks along a radix tree to share "
            "immutable key-value tensors across concurrent invocations with matching preambles. "
            "Subsequent requests hit resident HBM blocks directly and skip redundant transformer "
            "forward computation over shared prefix segments."
        ),
    ),
    CorpusChunk(
        chunk_id="vllm-flag-max-model-len",
        domain="llm-serving",
        title="Bounding Context Allocation with --max-model-len",
        text=(
            "vLLM validates startup KV cache capacity against the model config max_position_embeddings "
            "unless overridden via --max-model-len. On 24 GB GPUs, models shipping with 128K rotary "
            "metadata fail initialization until --max-model-len caps maximum sequence reservation "
            "to the workload's true 8K or 16K ceiling."
        ),
    ),
    CorpusChunk(
        chunk_id="cuda-oom-expandable-segments",
        domain="llm-serving",
        title="Virtual Address Pooling via PYTORCH_CUDA_ALLOC_CONF expandable_segments",
        text=(
            "Variable-length tensor workloads split and strand non-contiguous blocks inside the "
            "PyTorch GPU caching allocator, triggering CUDA out-of-memory failures despite gigabytes "
            "of free HBM. Setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True backs allocator "
            "arenas with stitchable virtual memory mappings that grow without contiguous pages."
        ),
    ),
    CorpusChunk(
        chunk_id="speculative-ngram-decoding",
        domain="llm-serving",
        title="Draft-Target Verification in Speculative Decoding",
        text=(
            "Speculative decoding proposes K candidate continuation tokens using a lightweight draft "
            "head or prompt n-gram lookup, then verifies all K candidates in one parallel target "
            "model forward pass. Memory-bandwidth-bound batch-1 generation accelerates by 1.6x to "
            "2.4x on structured code and JSON outputs without altering output distributions."
        ),
    ),
    # --- Domain 4: http-resilience (6 chunks) ---
    CorpusChunk(
        chunk_id="http-429-retry-after",
        domain="http-resilience",
        title="Respecting HTTP 429 Too Many Requests and Retry-After Headers",
        text=(
            "When an API gateway returns status HTTP 429, compliant SDK clients inspect the "
            "Retry-After response header defined in RFC 9110 before scheduling another attempt. "
            "Parsers must handle both non-negative integer delta-seconds values and IMF-fixdate "
            "timestamps, clamping maximum sleep intervals against caller request deadlines."
        ),
    ),
    CorpusChunk(
        chunk_id="http-jitter-circuit-breaker",
        domain="http-resilience",
        title="Decorrelated Backoff Jitter and Half-Open Canary Probing",
        text=(
            "Deterministic exponential sleep intervals cause fleets of stalled workers to wake "
            "in lockstep and slam convalescing upstream services with correlated traffic spikes. "
            "Decorrelated uniform randomization disperses retry schedules across time while "
            "half-open state machines gate full traffic restoration behind single canary probes."
        ),
    ),
    CorpusChunk(
        chunk_id="http-load-shedding-lifo",
        domain="http-resilience",
        title="LIFO Admission Queues and Deadline-Aware Load Shedding",
        text=(
            "Under saturation, first-in first-out request queues force every arriving packet to wait "
            "behind stale work whose upstream caller timeout has already elapsed. Switching admission "
            "control to last-in first-out ordering with active sojourn-time eviction serves recent "
            "arrivals immediately and sheds expired backlog items at minimal CPU cost."
        ),
    ),
    CorpusChunk(
        chunk_id="http-idempotency-key-ledger",
        domain="http-resilience",
        title="Atomic Mutation Deduplication with Idempotency-Key Headers",
        text=(
            "Payment and provisioning POST endpoints persist client-generated UUID Idempotency-Key "
            "headers inside a unique relational ledger alongside serialized response payloads. "
            "Network timeouts followed by automatic client replays return the committed outcome "
            "instead of double-charging or duplicating downstream side effects."
        ),
    ),
    CorpusChunk(
        chunk_id="envoy-flag-uo-overflow",
        domain="http-resilience",
        title="Envoy Access Log Response Flag UO Upstream Overflow",
        text=(
            "Envoy proxies emit response flag UO (UpstreamOverflow) alongside HTTP 503 when pending "
            "requests or active connections breach cluster circuit_breakers thresholds. Inspect "
            "upstream_rq_pending_overflow counters before raising max_pending_requests so overload "
            "does not cascade into downstream worker pools."
        ),
    ),
    CorpusChunk(
        chunk_id="envoy-flag-uf-connect",
        domain="http-resilience",
        title="Envoy Access Log Response Flag UF Upstream Connection Failure",
        text=(
            "Envoy proxies emit response flag UF (UpstreamConnectionFailure) alongside HTTP 503 when "
            "TCP SYN handshakes or mTLS transport sockets fail against selected upstream endpoints. "
            "Verify outlier_detection ejection health and confirm backend mTLS SAN certificates "
            "match cluster transport_socket configuration."
        ),
    ),
    # --- Domain 5: postgres-storage (6 chunks) ---
    CorpusChunk(
        chunk_id="pg-sqlstate-40p01",
        domain="postgres-storage",
        title="Remediating PostgreSQL SQLSTATE 40P01 deadlock_detected",
        text=(
            "PostgreSQL aborts a transaction with SQLSTATE 40P01 (deadlock_detected) after deadlock_timeout "
            "when two or more sessions hold row-level locks in a circular wait graph. Eliminate cyclic "
            "contention by sorting primary keys deterministically prior to multi-row UPDATE or SELECT "
            "FOR UPDATE statements across all application codepaths."
        ),
    ),
    CorpusChunk(
        chunk_id="pg-sqlstate-55p03",
        domain="postgres-storage",
        title="Handling PostgreSQL SQLSTATE 55P03 lock_not_available",
        text=(
            "Queries executed with FOR UPDATE NOWAIT or bounded lock_timeout fail immediately with "
            "SQLSTATE 55P03 (lock_not_available) rather than queuing behind long-running DDL or row "
            "locks. Online schema migrations should pair short lock_timeout retries with non-blocking "
            "CREATE INDEX CONCURRENTLY operations."
        ),
    ),
    CorpusChunk(
        chunk_id="pg-sqlstate-53300",
        domain="postgres-storage",
        title="Mitigating SQLSTATE 53300 too_many_connections with PgBouncer",
        text=(
            "PostgreSQL rejects new client handshakes with SQLSTATE 53300 (too_many_connections) once "
            "active backend processes reach max_connections minus superuser_reserved_connections. "
            "Place PgBouncer in transaction pooling mode ahead of the primary so thousands of stateless "
            "pods multiplex onto a compact pool of 100 to 200 server backends."
        ),
    ),
    CorpusChunk(
        chunk_id="pg-idle-in-transaction-25p03",
        domain="postgres-storage",
        title="Expiring Abandoned Snapshots Hold Back MVCC Horizons",
        text=(
            "Uncommitted sessions sitting in open transaction blocks pin old MVCC xmin horizons, "
            "blocking dead tuple cleanup and stalling ACCESS EXCLUSIVE catalog locks across tables. "
            "Configure idle_in_transaction_session_timeout to terminate leaked ORM transactions "
            "with SQLSTATE 25P03 and release held virtual snapshots."
        ),
    ),
    CorpusChunk(
        chunk_id="pg-autovacuum-hot-updates",
        domain="postgres-storage",
        title="MVCC Dead Tuple Reclamation and Heap-Only Tuple Pruning",
        text=(
            "High-churn MVCC tables accumulate obsolete row versions until background autovacuum "
            "workers scan heap pages and mark space reusable without locking concurrent DML writers. "
            "Lowering autovacuum_vacuum_scale_factor to 0.02 alongside fillfactor 85 enables "
            "in-page Heap-Only Tuple pruning on unindexed column updates."
        ),
    ),
    CorpusChunk(
        chunk_id="pg-wal-checkpoint-spikes",
        domain="postgres-storage",
        title="Smoothing PostgreSQL WAL Checkpoint I/O Bursts",
        text=(
            "Aggressive max_wal_size exhaustion forces frequent dirty buffer flushes that saturate "
            "NVMe write IOPS. Increasing max_wal_size while setting checkpoint_completion_target "
            "to 0.9 spreads dirty page writeback evenly across the checkpoint interval."
        ),
    ),
    # --- Domain 6: k8s-runtime (6 chunks) ---
    CorpusChunk(
        chunk_id="k8s-exit-137-oomkilled",
        domain="k8s-runtime",
        title="Container Termination Exit Code 137 and cgroup v2 memory.max",
        text=(
            "Kubernetes reports Reason OOMKilled with exit code 137 (128 + SIGKILL 9) when a pod's "
            "resident anonymous plus page-cache footprint breaches its cgroup v2 memory.max limit. "
            "Inspect container_memory_working_set_bytes alongside JVM or Python allocator arenas "
            "to separate true memory leaks from undersized pod limits."
        ),
    ),
    CorpusChunk(
        chunk_id="k8s-exit-143-sigterm",
        domain="k8s-runtime",
        title="Endpoint Slice Deregistration Propagation and Graceful Connection Draining",
        text=(
            "Pod deletion dispatches SIGTERM (exit code 143) simultaneously with asynchronous "
            "kube-proxy and ingress controller endpoint removal. Inserting a brief preStop sleep "
            "hook lets load-balancer control planes stop routing new TCP streams before the server "
            "closes its listening socket and drains in-flight work within terminationGracePeriodSeconds."
        ),
    ),
    CorpusChunk(
        chunk_id="k8s-coredns-ndots-five",
        domain="k8s-runtime",
        title="CoreDNS Query Amplification from Default ndots:5 Resolver Config",
        text=(
            "Default pod /etc/resolv.conf files ship with options ndots:5, causing external lookups "
            "such as api.stripe.com to issue four sequential NXDOMAIN search-domain queries across "
            "cluster.local suffixes over both IPv4 and IPv6 before resolving the FQDN. Append a "
            "trailing dot or lower dnsConfig ndots to 2 to cut CoreDNS packet rates by 70 percent."
        ),
    ),
    CorpusChunk(
        chunk_id="k8s-conntrack-table-full",
        domain="k8s-runtime",
        title="Node nf_conntrack Table Exhaustion and Packet Drops",
        text=(
            "High-connection NAT workloads log nf_conntrack: table full, dropping packet in kernel "
            "dmesg when tracked flows exceed net.netfilter.nf_conntrack_max on Kubernetes worker nodes. "
            "Enable HTTP keep-alive connection pooling to cluster services and scale conntrack buckets "
            "proportionally to node RAM."
        ),
    ),
    CorpusChunk(
        chunk_id="k8s-readiness-gate-warmup",
        domain="k8s-runtime",
        title="Separating startupProbe Model Loading from Liveness Restarts",
        text=(
            "ML serving containers that load multi-gigabyte safetensors weights over network storage "
            "often exceed aggressive livenessProbe windows and enter CrashLoopBackOff loops. Gate "
            "initial weight deserialization and CUDA graph capture behind a dedicated startupProbe "
            "so liveness checks activate only after warmup completes."
        ),
    ),
    CorpusChunk(
        chunk_id="k8s-exit-139-sigsegv",
        domain="k8s-runtime",
        title="Container Termination Exit Code 139 and SIGSEGV Core Dumps",
        text=(
            "Kubernetes reports container termination with exit code 139 (128 + SIGSEGV 11) when "
            "native C++, Rust, or custom CUDA Cython extensions dereference unaligned pointers or "
            "corrupt stack frames. Symbolicate core dumps against debuginfo artifacts and verify "
            "shared library ABI compatibility across manylinux base images."
        ),
    ),
    # --- Domain 7: vector-indexing (6 chunks) ---
    CorpusChunk(
        chunk_id="vec-hnsw-efsearch-tuning",
        domain="vector-indexing",
        title="Calibrating HNSW efSearch Beam Width vs Query Latency",
        text=(
            "In Hierarchical Navigable Small World (HNSW) graphs, efSearch controls the dynamic "
            "candidate priority queue size explored at query time while M and efConstruction remain "
            "fixed after index build. Raising efSearch from 32 to 128 lifts Recall@10 from roughly "
            "0.88 to 0.98 at sub-linear latency growth without rebuilding graph edges."
        ),
    ),
    CorpusChunk(
        chunk_id="vec-ivfpq-nprobe-codebook",
        domain="vector-indexing",
        title="Product Quantization Sub-Vector Codebooks in IVF Indexes",
        text=(
            "Inverted File with Product Quantization partitions high-dimensional embeddings into "
            "orthogonal sub-spaces and replaces 32-bit floats with compact 8-bit centroid codebook "
            "identifiers. Asymmetric distance lookup tables shrink resident DRAM footprints by 16x "
            "to 32x, trading mild quantization distortion that a stage-two reranker recovers."
        ),
    ),
    CorpusChunk(
        chunk_id="vec-scalar-int8-calibration",
        domain="vector-indexing",
        title="Scalar INT8 Quantization and Asymmetric Rescoring",
        text=(
            "Scalar quantization linearly maps per-dimension float32 min-max quantiles into signed "
            "int8 integers, cutting memory consumption by 4x while using AVX-512 VNNI dot-product "
            "instructions. Fetching 3x oversampled int8 candidates and rescoring against disk-backed "
            "float32 vectors preserves 99 percent of exact recall."
        ),
    ),
    CorpusChunk(
        chunk_id="vec-embedding-drift-mmd",
        domain="vector-indexing",
        title="Covariate Shift Detection Across Dense Representation Manifolds",
        text=(
            "Monitor live query and document representation manifolds using rolling two-sample Maximum "
            "Mean Discrepancy windows alongside centroid cosine distance tracking against reference "
            "baselines. Sustained distribution divergence warns when domain terminology evolves past "
            "the frozen encoder's pretraining vocabulary."
        ),
    ),
    CorpusChunk(
        chunk_id="vec-filtered-hnsw-prepost",
        domain="vector-indexing",
        title="Pre-Filtering vs Post-Filtering Selectivity Pitfalls in ANN Search",
        text=(
            "Applying strict tenant or timestamp predicates after fetching top-K HNSW neighbors yields "
            "empty result sets whenever matching records represent under one percent of the index. "
            "Switch to roaring-bitmap pre-filtering or navigable filtered graph traversal when "
            "predicate selectivity drops below five percent."
        ),
    ),
    CorpusChunk(
        chunk_id="vec-cosine-norm-pitfall",
        domain="vector-indexing",
        title="L2 Normalization Invariants for Inner-Product FAISS Indexes",
        text=(
            "FAISS IndexFlatIP computes raw unnormalized dot products rather than dividing by vector "
            "magnitudes. Unless both corpus and query embeddings are unit-L2-normalized prior to "
            "indexing and search, verbose passages with larger norms dominate top ranks regardless "
            "of directional cosine alignment."
        ),
    ),
    # --- Domain 8: linux-observability (6 chunks) ---
    CorpusChunk(
        chunk_id="linux-sysctl-tcp-tw-reuse",
        domain="linux-observability",
        title="Relieving Ephemeral Port Exhaustion with net.ipv4.tcp_tw_reuse",
        text=(
            "Reverse proxies opening thousands of short-lived upstream connections per second can "
            "exhaust net.ipv4.ip_local_port_range with sockets lingering in TIME_WAIT for 60 seconds. "
            "Enabling net.ipv4.tcp_tw_reuse=1 alongside TCP timestamps permits safe kernel reuse of "
            "outbound TIME_WAIT tuple slots when upstream keep-alive pools cannot be used."
        ),
    ),
    CorpusChunk(
        chunk_id="linux-sysctl-somaxconn",
        domain="linux-observability",
        title="Sizing SYN and Accept Backlogs via net.core.somaxconn",
        text=(
            "Linux silently clamps application listen(fd, backlog) arguments to net.core.somaxconn "
            "(legacy default 128 or 4096). During traffic spikes or brief GC pauses, accept queue "
            "overflows increment TcpExtListenOverflows in /proc/net/netstat and trigger client "
            "connection resets unless both somaxconn and tcp_max_syn_backlog are raised."
        ),
    ),
    CorpusChunk(
        chunk_id="otel-w3c-traceparent-header",
        domain="linux-observability",
        title="Propagating W3C Trace Context traceparent Headers Across Proxies",
        text=(
            "Distributed OpenTelemetry spans correlate across HTTP hops via the lowercase W3C "
            "traceparent header formatted as version-trace_id-parent_id-trace_flags, such as "
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01. API gateways that strip "
            "unknown headers split end-to-end call graphs into disconnected root traces."
        ),
    ),
    CorpusChunk(
        chunk_id="otel-tail-based-sampling",
        domain="linux-observability",
        title="Deferred Span Retention Policies in Collector Tiers",
        text=(
            "Head-based probabilistic sampling drops ninety-nine percent of anomalous traces before "
            "downstream outcomes are known. Collector-tier deferred retention buffers trace segments "
            "in memory keyed by trace ID until root completion, persisting spans selectively when "
            "status flags indicate faults or duration percentiles breach SLO ceilings."
        ),
    ),
    CorpusChunk(
        chunk_id="otel-baggage-cardinality",
        domain="linux-observability",
        title="Curbing TSDB Inverted Index Explosion from Unbounded Tag Dimensions",
        text=(
            "Attaching high-entropy request attributes such as session UUIDs, raw URLs, or account "
            "identifiers onto Prometheus or OpenTelemetry metric dimensions multiplies active series "
            "combinatorially in TSDB postings lists. Strip unbounded labels at collector processors "
            "and keep high-cardinality attributes strictly on structured log or span events."
        ),
    ),
    CorpusChunk(
        chunk_id="linux-ebpf-offcpu-profiling",
        domain="linux-observability",
        title="Tracing Scheduler Sleep Stacks with eBPF finish_task_switch",
        text=(
            "CPU flamegraphs sampled on hardware performance counters completely miss threads blocked "
            "on mutex contention, futex waits, or disk I/O. Attaching eBPF kprobes to kernel "
            "finish_task_switch captures off-CPU kernel and userspace stack traces weighted by "
            "microseconds spent descheduled."
        ),
    ),
)


EVAL_QUERIES: tuple[EvalQuery, ...] = (
    # --- Slice 1: exact_id (14 queries: error codes, SKUs, CLI flags, RFC headers, sysctls) ---
    EvalQuery(
        query_id="q_exact_01",
        query_slice="exact_id",
        query_text="ingress returning ERR_SSL_PROTOCOL_ERROR to clients",
        relevant_grades={"tls-alpn-record-mismatch": 2, "tls-sni-routing-mismatch": 1},
    ),
    EvalQuery(
        query_id="q_exact_02",
        query_slice="exact_id",
        query_text="aws g5.12xlarge node hardware specs and sharding guidance",
        relevant_grades={"gpu-sku-g5-12xlarge": 2, "gpu-sku-g5-2xlarge": 1},
    ),
    EvalQuery(
        query_id="q_exact_03",
        query_slice="exact_id",
        query_text="what does --max-num-batched-tokens control in production",
        relevant_grades={"vllm-flag-batched-tokens": 2, "vllm-flag-max-model-len": 1},
    ),
    EvalQuery(
        query_id="q_exact_04",
        query_slice="exact_id",
        query_text="HTTP 429 Retry-After header handling",
        relevant_grades={"http-429-retry-after": 2, "http-jitter-circuit-breaker": 1},
    ),
    EvalQuery(
        query_id="q_exact_05",
        query_slice="exact_id",
        query_text="postgres query failed with SQLSTATE 55P03",
        relevant_grades={"pg-sqlstate-55p03": 2, "pg-sqlstate-40p01": 1},
    ),
    EvalQuery(
        query_id="q_exact_06",
        query_slice="exact_id",
        query_text="envoy 503 error with response flag UO in access logs",
        relevant_grades={"envoy-flag-uo-overflow": 2, "envoy-flag-uf-connect": 1},
    ),
    EvalQuery(
        query_id="q_exact_07",
        query_slice="exact_id",
        query_text="kubernetes pod crashed with exit code 139",
        relevant_grades={"k8s-exit-139-sigsegv": 2, "k8s-exit-137-oomkilled": 1},
    ),
    EvalQuery(
        query_id="q_exact_08",
        query_slice="exact_id",
        query_text="sysctl net.ipv4.tcp_tw_reuse=1 setting",
        relevant_grades={"linux-sysctl-tcp-tw-reuse": 2, "linux-sysctl-somaxconn": 1},
    ),
    EvalQuery(
        query_id="q_exact_09",
        query_slice="exact_id",
        query_text="CUDA_LAUNCH_BLOCKING=1 flag",
        relevant_grades={"cuda-launch-blocking-debug": 2, "nccl-ib-disable-env": 1},
    ),
    EvalQuery(
        query_id="q_exact_10",
        query_slice="exact_id",
        query_text="postgres error code 25P03 on open session",
        relevant_grades={"pg-idle-in-transaction-25p03": 2, "pg-sqlstate-53300": 1},
    ),
    EvalQuery(
        query_id="q_exact_11",
        query_slice="exact_id",
        query_text="python service failing with X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT_LOCALLY",
        relevant_grades={"tls-x509-missing-intermediate": 2, "tls-mtls-cert-rotation": 1},
    ),
    EvalQuery(
        query_id="q_exact_12",
        query_slice="exact_id",
        query_text="pod latency spike caused by ndots:5 configuration",
        relevant_grades={"k8s-coredns-ndots-five": 2, "k8s-conntrack-table-full": 1},
    ),
    EvalQuery(
        query_id="q_exact_13",
        query_slice="exact_id",
        query_text="how parameter efSearch impacts query performance",
        relevant_grades={"vec-hnsw-efsearch-tuning": 2, "vec-filtered-hnsw-prepost": 1},
    ),
    EvalQuery(
        query_id="q_exact_14",
        query_slice="exact_id",
        query_text="TcpExtListenOverflows counter incrementing in /proc/net/netstat",
        relevant_grades={"linux-sysctl-somaxconn": 2, "linux-sysctl-tcp-tw-reuse": 1},
    ),
    # --- Slice 2: semantic_paraphrase (14 queries: conceptual / symptom vocabulary shifts) ---
    EvalQuery(
        query_id="q_sem_01",
        query_slice="semantic_paraphrase",
        query_text="preventing synchronized client retry storms from crushing a recovering backend",
        relevant_grades={"http-jitter-circuit-breaker": 2, "http-load-shedding-lifo": 1},
    ),
    EvalQuery(
        query_id="q_sem_02",
        query_slice="semantic_paraphrase",
        query_text="stopping huge document prompts from freezing token streaming for active chat users",
        relevant_grades={"vllm-chunked-prefill-sched": 2, "vllm-flag-batched-tokens": 1},
    ),
    EvalQuery(
        query_id="q_sem_03",
        query_slice="semantic_paraphrase",
        query_text="shrinking vector search RAM usage by compressing embeddings into tiny lookup codes",
        relevant_grades={"vec-ivfpq-nprobe-codebook": 2, "vec-scalar-int8-calibration": 1},
    ),
    EvalQuery(
        query_id="q_sem_04",
        query_slice="semantic_paraphrase",
        query_text="keeping distributed trace records only when a request fails or runs unusually slow",
        relevant_grades={"otel-tail-based-sampling": 2, "otel-w3c-traceparent-header": 1},
    ),
    EvalQuery(
        query_id="q_sem_05",
        query_slice="semantic_paraphrase",
        query_text="letting rolling pod shutdowns finish active HTTP responses before containers exit",
        relevant_grades={"k8s-exit-143-sigterm": 2, "k8s-readiness-gate-warmup": 1},
    ),
    EvalQuery(
        query_id="q_sem_06",
        query_slice="semantic_paraphrase",
        query_text="cleaning up obsolete row versions in busy relational tables without blocking writes",
        relevant_grades={"pg-autovacuum-hot-updates": 2, "pg-idle-in-transaction-25p03": 1},
    ),
    EvalQuery(
        query_id="q_sem_07",
        query_slice="semantic_paraphrase",
        query_text="serving newest waiting callers first and dropping timed-out requests during overload",
        relevant_grades={"http-load-shedding-lifo": 2, "http-jitter-circuit-breaker": 1},
    ),
    EvalQuery(
        query_id="q_sem_08",
        query_slice="semantic_paraphrase",
        query_text="reusing attention states across chat turns that share the same system prompt",
        relevant_grades={"vllm-prefix-kv-caching": 2, "vllm-chunked-prefill-sched": 1},
    ),
    EvalQuery(
        query_id="q_sem_09",
        query_slice="semantic_paraphrase",
        query_text="detecting when live production search queries shift away from our embedding model training data",
        relevant_grades={"vec-embedding-drift-mmd": 2, "vec-cosine-norm-pitfall": 1},
    ),
    EvalQuery(
        query_id="q_sem_10",
        query_slice="semantic_paraphrase",
        query_text="avoiding GPU out-of-memory crashes caused by fragmented memory blocks on variable lengths",
        relevant_grades={"cuda-oom-expandable-segments": 2, "vllm-flag-max-model-len": 1},
    ),
    EvalQuery(
        query_id="q_sem_11",
        query_slice="semantic_paraphrase",
        query_text="checking whether a certificate was revoked during login without calling outside servers live",
        relevant_grades={"tls-ocsp-stapling-timeout": 2, "tls-mtls-cert-rotation": 1},
    ),
    EvalQuery(
        query_id="q_sem_12",
        query_slice="semantic_paraphrase",
        query_text="moving tensors directly between graphics cards in one server without detouring through CPU RAM",
        relevant_grades={"nccl-p2p-nvl-ring": 2, "gpu-sku-p4d-24xlarge": 1},
    ),
    EvalQuery(
        query_id="q_sem_13",
        query_slice="semantic_paraphrase",
        query_text="preventing unique user IDs in monitoring labels from blowing up metrics database memory",
        relevant_grades={"otel-baggage-cardinality": 2, "otel-tail-based-sampling": 1},
    ),
    EvalQuery(
        query_id="q_sem_14",
        query_slice="semantic_paraphrase",
        query_text="stopping forgotten uncommitted database sessions from blocking schema migrations",
        relevant_grades={"pg-idle-in-transaction-25p03": 2, "pg-sqlstate-55p03": 1},
    ),
)


def load_corpus() -> tuple[CorpusChunk, ...]:
    """Return the immutable 48-chunk engineering corpus."""
    return CORPUS_CHUNKS


def load_eval_queries() -> tuple[EvalQuery, ...]:
    """Return the immutable 28-query diagnostic evaluation suite."""
    return EVAL_QUERIES
