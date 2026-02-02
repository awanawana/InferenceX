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
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_queries: int = 0
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
            r'vllm:kv_cache_usage_perc\{[^}]*\}\s+([\d.e+-]+)'
        )

        # Running/waiting requests
        snapshot.num_requests_running = int(get_value(
            r'vllm:num_requests_running\{[^}]*\}\s+([\d.e+-]+)'
        ))
        snapshot.num_requests_waiting = int(get_value(
            r'vllm:num_requests_waiting\{[^}]*\}\s+([\d.e+-]+)'
        ))

        # Prefix cache (cumulative counters)
        snapshot.prefix_cache_hits = int(get_value(
            r'vllm:prefix_cache_hits_total\{[^}]*\}\s+([\d.e+-]+)'
        ))
        snapshot.prefix_cache_queries = int(get_value(
            r'vllm:prefix_cache_queries_total\{[^}]*\}\s+([\d.e+-]+)'
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

    def generate_plots(self, output_prefix: str = "metrics") -> None:
        """Generate visualization plots from collected metrics."""
        if len(self.snapshots) < 2:
            print("Not enough data points for plots")
            return

        # Convert to relative time (seconds from start)
        start_time = self.snapshots[0].timestamp
        times = [(s.timestamp - start_time) for s in self.snapshots]

        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle("vLLM Server Metrics During Benchmark", fontsize=14)

        # 1. KV Cache Usage vs Time
        ax = axes[0, 0]
        kv_usage = [s.kv_cache_usage * 100 for s in self.snapshots]
        ax.plot(times, kv_usage, 'b-', linewidth=1.5)
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
        for i in range(1, len(self.snapshots)):
            delta_hits = self.snapshots[i].prefix_cache_hits - self.snapshots[i-1].prefix_cache_hits
            delta_queries = self.snapshots[i].prefix_cache_queries - self.snapshots[i-1].prefix_cache_queries
            if delta_queries > 0:
                hit_rates.append(100.0 * delta_hits / delta_queries)
            else:
                hit_rates.append(hit_rates[-1] if hit_rates else 0)
        ax.plot(times[1:], hit_rates, 'purple', linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cache Hit Rate (%)")
        ax.set_title("Prefix Cache Hit Rate (Rolling)")
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
            print(f"Overall cache hit rate: {hit_rate:.1f}%")
            print(f"  - Cache hits: {delta_hits:,} tokens")
            print(f"  - Cache queries: {delta_queries:,} tokens")

        print("="*60 + "\n")
