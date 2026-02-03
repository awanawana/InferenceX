"""
Metrics collector for vLLM server during benchmarks.
Polls /metrics endpoint and generates visualizations.
"""

import asyncio
import re
import time
from dataclasses import dataclass, field

import aiohttp
import matplotlib.pyplot as plt


@dataclass
class MetricsSnapshot:
    timestamp: float
    kv_cache_usage: float = 0.0
    cpu_kv_cache_usage: float = 0.0
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_queries: int = 0
    cpu_prefix_cache_hits: int = 0
    cpu_prefix_cache_queries: int = 0
    prompt_tokens: int = 0
    generation_tokens: int = 0
    num_preemptions: int = 0
    request_success: int = 0


@dataclass
class MetricsCollector:
    base_url: str
    poll_interval: float = 1.0
    snapshots: list[MetricsSnapshot] = field(default_factory=list)
    _running: bool = False
    _task: asyncio.Task | None = None

    def _parse_metrics(self, text: str) -> MetricsSnapshot:
        """Parse Prometheus metrics text format."""
        snapshot = MetricsSnapshot(timestamp=time.time())

        # Helper to extract gauge/counter value
        def get_value(pattern: str, default: float = 0.0) -> float:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
            return default

        # KV cache usage (0-1 scale)
        snapshot.kv_cache_usage = get_value(
            r'vllm:gpu_cache_usage_perc\{[^}]*\}\s+([\d.e+-]+)'
        )
        # Fallback to old metric name if new one not found
        if snapshot.kv_cache_usage == 0.0:
            snapshot.kv_cache_usage = get_value(
                r'vllm:kv_cache_usage_perc\{[^}]*\}\s+([\d.e+-]+)'
            )

        # CPU/offloaded KV cache usage
        snapshot.cpu_kv_cache_usage = get_value(
            r'vllm:cpu_cache_usage_perc\{[^}]*\}\s+([\d.e+-]+)'
        )

        # Running/waiting requests
        snapshot.num_requests_running = int(get_value(
            r'vllm:num_requests_running\{[^}]*\}\s+([\d.e+-]+)'
        ))
        snapshot.num_requests_waiting = int(get_value(
            r'vllm:num_requests_waiting\{[^}]*\}\s+([\d.e+-]+)'
        ))

        # Prefix cache (cumulative counters) - GPU
        snapshot.prefix_cache_hits = int(get_value(
            r'vllm:prefix_cache_hits_total\{[^}]*\}\s+([\d.e+-]+)'
        ))
        snapshot.prefix_cache_queries = int(get_value(
            r'vllm:prefix_cache_queries_total\{[^}]*\}\s+([\d.e+-]+)'
        ))

        # Prefix cache - external/offloaded (KV connector cross-instance cache)
        snapshot.cpu_prefix_cache_hits = int(get_value(
            r'vllm:external_prefix_cache_hits_total\{[^}]*\}\s+([\d.e+-]+)'
        ))
        snapshot.cpu_prefix_cache_queries = int(get_value(
            r'vllm:external_prefix_cache_queries_total\{[^}]*\}\s+([\d.e+-]+)'
        ))

        # Token counters
        snapshot.prompt_tokens = int(get_value(
            r'vllm:prompt_tokens_total\{[^}]*\}\s+([\d.e+-]+)'
        ))
        snapshot.generation_tokens = int(get_value(
            r'vllm:generation_tokens_total\{[^}]*\}\s+([\d.e+-]+)'
        ))

        # Preemptions
        snapshot.num_preemptions = int(get_value(
            r'vllm:num_preemptions_total\{[^}]*\}\s+([\d.e+-]+)'
        ))

        # Request success (sum all finish reasons)
        for match in re.finditer(
            r'vllm:request_success_total\{[^}]*finished_reason="[^"]*"[^}]*\}\s+([\d.e+-]+)',
            text
        ):
            snapshot.request_success += int(float(match.group(1)))

        return snapshot

    async def _poll_loop(self) -> None:
        """Background polling loop."""
        metrics_url = f"{self.base_url}/metrics"
        async with aiohttp.ClientSession() as session:
            while self._running:
                try:
                    async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            snapshot = self._parse_metrics(text)
                            self.snapshots.append(snapshot)
                except Exception as e:
                    print(f"Metrics poll error: {e}")

                await asyncio.sleep(self.poll_interval)

    def start(self) -> None:
        """Start background metrics collection."""
        if self._running:
            return
        self._running = True
        self.snapshots = []
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop metrics collection."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def generate_plots(
        self,
        output_prefix: str = "metrics",
        client_metrics: list | None = None,
    ) -> None:
        """Generate visualization plots from collected metrics.

        Args:
            output_prefix: Prefix for output file names
            client_metrics: Optional list of RequestStats from benchmark clients
        """
        if len(self.snapshots) < 2:
            print("Not enough data points for plots")
            return

        # Convert to relative time (seconds from start)
        start_time = self.snapshots[0].timestamp
        times = [(s.timestamp - start_time) for s in self.snapshots]

        # Create figure with subplots
        num_rows = 5 if client_metrics else 3
        fig, axes = plt.subplots(num_rows, 2, figsize=(14, 4 * num_rows))
        fig.suptitle("vLLM Server Metrics During Benchmark", fontsize=14)

        # 1. KV Cache Usage vs Time
        ax = axes[0, 0]
        kv_usage = [s.kv_cache_usage * 100 for s in self.snapshots]
        ax.plot(times, kv_usage, 'b-', label='GPU', linewidth=1.5)
        # Add external cache if available
        cpu_kv_usage = [s.cpu_kv_cache_usage * 100 for s in self.snapshots]
        if any(v > 0 for v in cpu_kv_usage):
            ax.plot(times, cpu_kv_usage, 'r--', label='External', linewidth=1.5)
            ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("KV Cache Usage (%)")
        ax.set_title("KV Cache Utilization Over Time")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        # 2. Running & Waiting Requests vs Time
        ax = axes[0, 1]
        running = [s.num_requests_running for s in self.snapshots]
        waiting = [s.num_requests_waiting for s in self.snapshots]
        ax.plot(times, running, 'g-', label='Running', linewidth=1.5)
        ax.plot(times, waiting, 'r-', label='Waiting', linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Requests")
        ax.set_title("Request Queue Depth")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Cache Hit Rate vs Time (computed from deltas)
        ax = axes[1, 0]
        hit_rates = []
        cpu_hit_rates = []
        has_cpu_cache = any(s.cpu_prefix_cache_queries > 0 for s in self.snapshots)
        for i in range(1, len(self.snapshots)):
            # GPU cache hit rate
            delta_hits = self.snapshots[i].prefix_cache_hits - self.snapshots[i-1].prefix_cache_hits
            delta_queries = self.snapshots[i].prefix_cache_queries - self.snapshots[i-1].prefix_cache_queries
            if delta_queries > 0:
                hit_rates.append(100.0 * delta_hits / delta_queries)
            else:
                hit_rates.append(hit_rates[-1] if hit_rates else 0)
            # External cache hit rate
            if has_cpu_cache:
                cpu_delta_hits = self.snapshots[i].cpu_prefix_cache_hits - self.snapshots[i-1].cpu_prefix_cache_hits
                cpu_delta_queries = self.snapshots[i].cpu_prefix_cache_queries - self.snapshots[i-1].cpu_prefix_cache_queries
                if cpu_delta_queries > 0:
                    cpu_hit_rates.append(100.0 * cpu_delta_hits / cpu_delta_queries)
                else:
                    cpu_hit_rates.append(cpu_hit_rates[-1] if cpu_hit_rates else 0)

        # Scatter plot for GPU cache hit rate
        ax.scatter(times[1:], hit_rates, alpha=0.3, s=5, c='purple', label='HBM')
        # Rolling average for GPU
        window = min(50, len(hit_rates) // 10) if len(hit_rates) > 10 else 1
        if window > 1:
            rolling_gpu = [
                sum(hit_rates[max(0, i - window):i + 1]) / len(hit_rates[max(0, i - window):i + 1])
                for i in range(len(hit_rates))
            ]
            ax.plot(times[1:], rolling_gpu, 'purple', linewidth=1.5, label=f'HBM avg (n={window})')

        # Scatter plot and rolling average for external cache
        if has_cpu_cache and cpu_hit_rates:
            ax.scatter(times[1:], cpu_hit_rates, alpha=0.3, s=5, c='orange', label='External')
            if window > 1:
                rolling_ext = [
                    sum(cpu_hit_rates[max(0, i - window):i + 1]) / len(cpu_hit_rates[max(0, i - window):i + 1])
                    for i in range(len(cpu_hit_rates))
                ]
                ax.plot(times[1:], rolling_ext, 'orange', linewidth=1.5, label=f'External avg (n={window})')

        ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cache Hit Rate (%)")
        ax.set_title("Prefix Cache Hit Rate")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        # 4. Throughput vs Time (tokens/sec)
        ax = axes[1, 1]
        throughputs = []
        for i in range(1, len(self.snapshots)):
            delta_gen = self.snapshots[i].generation_tokens - self.snapshots[i-1].generation_tokens
            delta_time = self.snapshots[i].timestamp - self.snapshots[i-1].timestamp
            if delta_time > 0:
                throughputs.append(delta_gen / delta_time)
            else:
                throughputs.append(0)
        ax.plot(times[1:], throughputs, 'orange', linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Tokens/sec")
        ax.set_title("Generation Throughput")
        ax.grid(True, alpha=0.3)

        # 5. Cumulative Cache Hit Rate vs Requests
        ax = axes[2, 0]
        requests = [s.request_success for s in self.snapshots]
        cumulative_hit_rate = []
        for s in self.snapshots:
            if s.prefix_cache_queries > 0:
                cumulative_hit_rate.append(100.0 * s.prefix_cache_hits / s.prefix_cache_queries)
            else:
                cumulative_hit_rate.append(0)
        ax.plot(requests, cumulative_hit_rate, 'teal', linewidth=1.5)
        ax.set_xlabel("Completed Requests")
        ax.set_ylabel("Cumulative Hit Rate (%)")
        ax.set_title("Cache Hit Rate vs Completed Requests")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        # 6. Throughput vs KV Cache Utilization (scatter)
        ax = axes[2, 1]
        # Use throughput from slot 4
        kv_for_scatter = [s.kv_cache_usage * 100 for s in self.snapshots[1:]]
        ax.scatter(kv_for_scatter, throughputs, alpha=0.5, s=10, c='coral')
        ax.set_xlabel("KV Cache Usage (%)")
        ax.set_ylabel("Throughput (tokens/sec)")
        ax.set_title("Throughput vs KV Cache Utilization")
        ax.grid(True, alpha=0.3)

        # 7 & 8. Client metrics plots (TTFT and Latency vs Time)
        if client_metrics and len(client_metrics) > 0:
            # Sort by start time
            sorted_metrics = sorted(client_metrics, key=lambda x: x.start_time_ms)
            # Convert to relative time (seconds from first request)
            first_start = sorted_metrics[0].start_time_ms
            request_times = [(m.start_time_ms - first_start) / 1000.0 for m in sorted_metrics]
            ttfts = [m.ttft_ms for m in sorted_metrics]
            latencies = [m.latency_ms for m in sorted_metrics]

            # 7. TTFT vs Time
            ax = axes[3, 0]
            ax.scatter(request_times, ttfts, alpha=0.3, s=5, c='blue')
            # Add rolling average
            window = min(50, len(ttfts) // 10) if len(ttfts) > 10 else 1
            if window > 1:
                rolling_ttft = [
                    sum(ttfts[max(0, i - window):i + 1]) / len(ttfts[max(0, i - window):i + 1])
                    for i in range(len(ttfts))
                ]
                ax.plot(request_times, rolling_ttft, 'r-', linewidth=1.5, label=f'Rolling avg (n={window})')
                ax.legend()
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("TTFT (ms)")
            ax.set_title("Time to First Token vs Time")
            ax.grid(True, alpha=0.3)

            # 8. Latency vs Time
            ax = axes[3, 1]
            ax.scatter(request_times, latencies, alpha=0.3, s=5, c='green')
            # Add rolling average
            if window > 1:
                rolling_latency = [
                    sum(latencies[max(0, i - window):i + 1]) / len(latencies[max(0, i - window):i + 1])
                    for i in range(len(latencies))
                ]
                ax.plot(request_times, rolling_latency, 'r-', linewidth=1.5, label=f'Rolling avg (n={window})')
                ax.legend()
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Latency (ms)")
            ax.set_title("Request Latency vs Time")
            ax.grid(True, alpha=0.3)

            # 9. Interactivity (1/TPOT = tokens/sec) vs Time
            ax = axes[4, 0]
            # Filter out zero TPOT values to avoid division by zero
            tpots = [m.tpot_ms for m in sorted_metrics]
            interactivity = [1000.0 / t if t > 0 else 0 for t in tpots]  # Convert to tokens/sec
            ax.scatter(request_times, interactivity, alpha=0.3, s=5, c='purple')
            # Add rolling average
            if window > 1:
                rolling_inter = [
                    sum(interactivity[max(0, i - window):i + 1]) / len(interactivity[max(0, i - window):i + 1])
                    for i in range(len(interactivity))
                ]
                ax.plot(request_times, rolling_inter, 'r-', linewidth=1.5, label=f'Rolling avg (n={window})')
                ax.legend()
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Interactivity (tokens/sec)")
            ax.set_title("Decode Speed (1/TPOT) vs Time")
            ax.grid(True, alpha=0.3)

            # 10. TPOT vs Time (raw)
            ax = axes[4, 1]
            ax.scatter(request_times, tpots, alpha=0.3, s=5, c='orange')
            if window > 1:
                rolling_tpot = [
                    sum(tpots[max(0, i - window):i + 1]) / len(tpots[max(0, i - window):i + 1])
                    for i in range(len(tpots))
                ]
                ax.plot(request_times, rolling_tpot, 'r-', linewidth=1.5, label=f'Rolling avg (n={window})')
                ax.legend()
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("TPOT (ms)")
            ax.set_title("Time Per Output Token vs Time")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_prefix}_plots.png", dpi=150)
        print(f"Saved plots to {output_prefix}_plots.png")
        plt.close()

        # Also generate a summary
        self._print_summary()

    def _print_summary(self) -> None:
        """Print summary statistics."""
        if len(self.snapshots) < 2:
            return

        duration = self.snapshots[-1].timestamp - self.snapshots[0].timestamp
        total_gen_tokens = self.snapshots[-1].generation_tokens - self.snapshots[0].generation_tokens
        total_prompt_tokens = self.snapshots[-1].prompt_tokens - self.snapshots[0].prompt_tokens

        final = self.snapshots[-1]
        initial = self.snapshots[0]

        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)
        print(f"Duration: {duration:.1f}s")
        print(f"Total prompt tokens: {total_prompt_tokens:,}")
        print(f"Total generation tokens: {total_gen_tokens:,}")
        print(f"Avg generation throughput: {total_gen_tokens/duration:.1f} tok/s")
        print(f"Peak KV cache usage: {max(s.kv_cache_usage for s in self.snapshots)*100:.1f}%")
        print(f"Peak running requests: {max(s.num_requests_running for s in self.snapshots)}")
        print(f"Peak waiting requests: {max(s.num_requests_waiting for s in self.snapshots)}")
        print(f"Total preemptions: {final.num_preemptions - initial.num_preemptions}")

        if final.prefix_cache_queries > initial.prefix_cache_queries:
            delta_hits = final.prefix_cache_hits - initial.prefix_cache_hits
            delta_queries = final.prefix_cache_queries - initial.prefix_cache_queries
            hit_rate = 100.0 * delta_hits / delta_queries
            print(f"Overall GPU cache hit rate: {hit_rate:.1f}%")
            print(f"  - Cache hits: {delta_hits:,} tokens")
            print(f"  - Cache queries: {delta_queries:,} tokens")

        # External/offloaded cache stats if available
        if final.cpu_prefix_cache_queries > initial.cpu_prefix_cache_queries:
            cpu_delta_hits = final.cpu_prefix_cache_hits - initial.cpu_prefix_cache_hits
            cpu_delta_queries = final.cpu_prefix_cache_queries - initial.cpu_prefix_cache_queries
            cpu_hit_rate = 100.0 * cpu_delta_hits / cpu_delta_queries
            print(f"Overall external cache hit rate: {cpu_hit_rate:.1f}%")
            print(f"  - Cache hits: {cpu_delta_hits:,} tokens")
            print(f"  - Cache queries: {cpu_delta_queries:,} tokens")

        print("="*60 + "\n")
