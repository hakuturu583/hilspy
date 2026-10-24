//! The host's load as observable gauges (`sysinfo`), sampled when the
//! reader collects.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use opentelemetry::metrics::Meter;
use opentelemetry::KeyValue;
use sysinfo::{
    CpuRefreshKind, MemoryRefreshKind, Pid, ProcessRefreshKind, ProcessesToUpdate, RefreshKind,
    System,
};

/// One refresh serves every gauge of a collection.
const MIN_REFRESH: Duration = Duration::from_millis(200);

#[derive(Default)]
struct Sample {
    /// Fraction of all CPUs (0..=1).
    process_cpu: f64,
    process_memory: u64,
    cpus: Vec<f64>,
    memory: f64,
    load_1m: f64,
}

struct Sampler {
    system: System,
    pid: Option<Pid>,
    at: Option<Instant>,
    sample: Arc<Sample>,
}

impl Sampler {
    fn new() -> Self {
        let system = System::new_with_specifics(
            RefreshKind::nothing()
                .with_cpu(CpuRefreshKind::nothing().with_cpu_usage())
                .with_memory(MemoryRefreshKind::nothing().with_ram()),
        );
        Self {
            system,
            pid: sysinfo::get_current_pid().ok(),
            at: None,
            sample: Arc::default(),
        }
    }

    /// CPU usage is measured between two refreshes, so the first
    /// collection reads zero for it.
    fn sample(&mut self) -> Arc<Sample> {
        if self.at.is_some_and(|t| t.elapsed() < MIN_REFRESH) {
            return Arc::clone(&self.sample);
        }
        self.at = Some(Instant::now());
        let s = &mut self.system;
        s.refresh_cpu_usage();
        s.refresh_memory_specifics(MemoryRefreshKind::nothing().with_ram());
        let cpus: Vec<f64> = s
            .cpus()
            .iter()
            .map(|c| f64::from(c.cpu_usage()) / 100.0)
            .collect();
        let (mut process_cpu, mut process_memory) = (0.0, 0);
        if let Some(pid) = self.pid {
            s.refresh_processes_specifics(
                ProcessesToUpdate::Some(&[pid]),
                true,
                ProcessRefreshKind::nothing().with_cpu().with_memory(),
            );
            if let Some(p) = s.process(pid) {
                // sysinfo: 100 per fully used CPU.
                process_cpu = f64::from(p.cpu_usage()) / 100.0 / cpus.len().max(1) as f64;
                process_memory = p.memory();
            }
        }
        let total = s.total_memory();
        self.sample = Arc::new(Sample {
            process_cpu,
            process_memory,
            cpus,
            memory: if total == 0 {
                0.0
            } else {
                s.used_memory() as f64 / total as f64
            },
            load_1m: System::load_average().one,
        });
        Arc::clone(&self.sample)
    }
}

/// Register the host gauges on `meter`: how loaded the computer running
/// the simulation (the ECU it plays) is. [`super::init_from_env`] does this
/// for the provider it installs; call it for a provider of your own.
pub fn register_host_metrics(meter: &Meter) {
    let sampler = Arc::new(Mutex::new(Sampler::new()));
    let sample = move || sampler.lock().unwrap().sample();

    let s = sample.clone();
    meter
        .f64_observable_gauge("process.cpu.utilization")
        .with_unit("1")
        .with_description("CPU time of this process over the last interval, a fraction of all CPUs")
        .with_callback(move |o| o.observe(s().process_cpu, &[]))
        .build();
    let s = sample.clone();
    meter
        .u64_observable_gauge("process.memory.usage")
        .with_unit("By")
        .with_description("Resident memory of this process")
        .with_callback(move |o| o.observe(s().process_memory, &[]))
        .build();
    let s = sample.clone();
    meter
        .f64_observable_gauge("system.cpu.utilization")
        .with_unit("1")
        .with_description("Busy fraction of each logical CPU over the last interval")
        .with_callback(move |o| {
            for (n, u) in s().cpus.iter().enumerate() {
                o.observe(*u, &[KeyValue::new("cpu.logical_number", n as i64)]);
            }
        })
        .build();
    let s = sample.clone();
    meter
        .f64_observable_gauge("system.memory.utilization")
        .with_unit("1")
        .with_description("Used fraction of the host's memory")
        .with_callback(move |o| o.observe(s().memory, &[]))
        .build();
    meter
        .f64_observable_gauge("system.cpu.load_average.1m")
        .with_unit("{thread}")
        .with_description("Load average over one minute (0 on Windows)")
        .with_callback(move |o| o.observe(sample().load_1m, &[]))
        .build();
}
