//! IOKit / GPU-util governor (port of `nvml_gov` public interface).
//!
//! Static util ceiling from config; optional background poll when yielding.
//! When yielding and observed util > 90%, the session loop inserts a brief
//! pause so sibling GPU users get time slices.
//!
//! # Sensor source
//!
//! Reads Apple GPU device utilization via IOKit: walks every `IOAccelerator`
//! service, reads its `PerformanceStatistics` CFDictionary, and takes the max
//! `"Device Utilization %"` value across services (a Mac normally has one).
//! This is a direct Rust port of the `ctypes` IOKit binding in
//! `GPU/macos_sensors.py::_query_iokit_gpu_utilization` — same service name,
//! same dictionary keys, same "return 0 on any failure" contract. Keeps the
//! **identical public interface** as `nvml_gov::UtilGovernor` so `session.rs`
//! calls it unchanged.
//!
//! This crate does not use the private/undocumented IOReport channel-sampling
//! path (`GPU/macos_sensors.py`'s `_ioreport_residency`, which is a stub
//! there too) — only the documented IOKit `IOAccelerator` service query.

use core_foundation_sys::base::{CFGetTypeID, CFRelease, CFTypeRef};
use core_foundation_sys::dictionary::{
    CFDictionaryGetTypeID, CFDictionaryGetValue, CFDictionaryRef, CFMutableDictionaryRef,
};
use core_foundation_sys::number::{
    kCFNumberSInt64Type, CFNumberGetTypeID, CFNumberGetValue, CFNumberRef,
};
use core_foundation_sys::string::{kCFStringEncodingUTF8, CFStringCreateWithCString, CFStringRef};
use io_kit_sys::types::io_iterator_t;
use io_kit_sys::{
    kIOMasterPortDefault, IOIteratorNext, IOObjectRelease, IORegistryEntryCreateCFProperties,
    IOServiceGetMatchingServices, IOServiceMatching,
};
use std::ffi::{c_void, CString};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Reconfigurable governor knobs plus the latest util sample, shared with the
/// poll thread.
#[derive(Debug)]
struct Knobs {
    /// Util ceiling 1–100; throttle fires above it when yielding.
    ceiling: AtomicU32,
    yielding: AtomicBool,
    /// Last GPU util percent 0–100 (0 while not yielding).
    last_util: AtomicU32,
    /// GPU-busy microseconds reported since the last poll tick, accumulated by
    /// the streaming loop and drained by [`poll_loop`].
    busy_us: AtomicU64,
    /// Our own share of the device, percent 0–100, over the last poll window.
    self_util: AtomicU32,
    /// `last_util - self_util`, floored at 0: load we did not cause.
    external_util: AtomicU32,
    stop: AtomicBool,
}

/// Shared utilization sample and reconfigurable governor knobs.
#[derive(Debug)]
pub struct UtilGovernor {
    knobs: Arc<Knobs>,
    handle: Option<JoinHandle<()>>,
}

impl UtilGovernor {
    /// Start the util poller. Values come from the CLI; `Configure` may later
    /// override them via [`reconfigure`](Self::reconfigure). The poll thread
    /// runs regardless of `yielding` (so a later `false -> true` override starts
    /// sampling with no thread churn) but only records util while yielding.
    ///
    /// Falls back to a silent no-op if sensors are unavailable (miner still
    /// runs; util stays 0 and throttle never fires).
    pub fn start(device_index: u32, utilization_ceiling: u32, yielding: bool) -> Self {
        let knobs = Arc::new(Knobs {
            ceiling: AtomicU32::new(utilization_ceiling.clamp(1, 100)),
            yielding: AtomicBool::new(yielding),
            last_util: AtomicU32::new(0),
            busy_us: AtomicU64::new(0),
            self_util: AtomicU32::new(0),
            external_util: AtomicU32::new(0),
            stop: AtomicBool::new(false),
        });
        let knobs_thread = Arc::clone(&knobs);
        let handle = Some(thread::spawn(move || {
            poll_loop(device_index, &knobs_thread)
        }));
        Self { knobs, handle }
    }

    /// A governor with a fixed sample and no poll thread: whole-device
    /// utilization `utilization` and our own share `self_utilization`, both in
    /// percent. For tests of throttle decisions.
    #[doc(hidden)]
    #[must_use]
    pub fn with_sample_for_test(
        utilization_ceiling: u32,
        yielding: bool,
        utilization: u32,
        self_utilization: u32,
    ) -> Self {
        let knobs = Arc::new(Knobs {
            ceiling: AtomicU32::new(utilization_ceiling.clamp(1, 100)),
            yielding: AtomicBool::new(yielding),
            last_util: AtomicU32::new(utilization),
            busy_us: AtomicU64::new(0),
            self_util: AtomicU32::new(self_utilization),
            external_util: AtomicU32::new(utilization.saturating_sub(self_utilization)),
            stop: AtomicBool::new(false),
        });
        Self {
            knobs,
            handle: None,
        }
    }

    /// Override the ceiling and yielding flag at runtime (config over CLI).
    pub fn reconfigure(&self, utilization_ceiling: u32, yielding: bool) {
        self.knobs
            .ceiling
            .store(utilization_ceiling.clamp(1, 100), Ordering::Relaxed);
        self.knobs.yielding.store(yielding, Ordering::Relaxed);
    }

    /// Current ceiling (CLI value, or the config override once applied).
    pub fn utilization_ceiling(&self) -> u32 {
        self.knobs.ceiling.load(Ordering::Relaxed)
    }

    /// Current yielding flag (CLI value, or the config override once applied).
    pub fn yielding(&self) -> bool {
        self.knobs.yielding.load(Ordering::Relaxed)
    }

    /// Last GPU util percent (0–100), or 0 if not yielding / unavailable.
    pub fn utilization(&self) -> f32 {
        self.knobs.last_util.load(Ordering::Relaxed) as f32
    }

    /// Our own share of the GPU (percent 0–100) over the last poll window.
    ///
    /// Computed from the GPU-busy microseconds the streaming loop reports via
    /// [`record_gpu_busy_us`](Self::record_gpu_busy_us), not from any sensor.
    pub fn self_utilization(&self) -> f32 {
        self.knobs.self_util.load(Ordering::Relaxed) as f32
    }

    /// Device load we did not cause (percent 0–100): `utilization` minus
    /// `self_utilization`, floored at 0.
    ///
    /// This is the contention signal. The raw sensor cannot provide it: it
    /// reports whole-device load, and the miner is normally the dominant user,
    /// so a high reading says nothing about whether anyone else wants the GPU.
    /// Subtracting our own measured contribution is what makes the remainder
    /// meaningful — and it needs no per-process data, which macOS does not
    /// publish anyway (there is no per-client GPU-time or threadgroup count in
    /// the IORegistry; `CommandQueueCount` counts queues every idle GUI app
    /// holds open, and `AGCInfo -> fLastSubmissionPID` is a single scalar).
    pub fn external_utilization(&self) -> f32 {
        self.knobs.external_util.load(Ordering::Relaxed) as f32
    }

    /// Report GPU-busy microseconds from a completed batch.
    ///
    /// Accumulates until the next poll tick converts it into
    /// [`self_utilization`](Self::self_utilization). Callers pass device time
    /// (`GPUEndTime - GPUStartTime`), never wall clock.
    pub fn record_gpu_busy_us(&self, us: u64) {
        let _ = self.knobs.busy_us.fetch_add(us, Ordering::Relaxed);
    }

    /// Fraction of the full threadgroup budget this miner should dispatch,
    /// in `0.005..=1.0`.
    ///
    /// Two independent factors:
    ///
    /// 1. **The ceiling itself.** `--utilization 80` means "aim to occupy 80% of
    ///    the GPU", so the budget is 80% of nominal. At the default 100 this is
    ///    1.0 and costs nothing.
    /// 2. **External pressure, only while yielding.** Each point of
    ///    [`external_utilization`](Self::external_utilization) gives back a point
    ///    of ceiling — but never more than half of it. A ceiling of 80 therefore
    ///    ranges over 80% → 40% of nominal, and 100 over 100% → 50%.
    ///
    /// The half-ceiling floor is deliberate: yielding is meant to share the
    /// device, not to surrender it. Without a floor, sustained external load
    /// (a video call, a compile) would drive the budget toward zero and stall
    /// mining entirely for as long as the other app ran.
    ///
    /// # Examples
    ///
    /// Pure once a governor exists: no Metal device is opened. Ceiling alone
    /// sets the scale while yielding is off.
    ///
    /// ```
    /// use quip_solver_metal::iokit_gov::UtilGovernor;
    ///
    /// let mut gov = UtilGovernor::start(0, 80, false);
    /// assert!((gov.budget_scale() - 0.8).abs() < 1e-9);
    /// assert_eq!(gov.utilization_ceiling(), 80);
    /// assert!(!gov.yielding());
    /// gov.reconfigure(100, false);
    /// assert!((gov.budget_scale() - 1.0).abs() < f64::EPSILON);
    /// gov.stop();
    /// ```
    pub fn budget_scale(&self) -> f64 {
        let ceiling = f64::from(self.knobs.ceiling.load(Ordering::Relaxed));
        let headroom = if self.knobs.yielding.load(Ordering::Relaxed) {
            let external = f64::from(self.knobs.external_util.load(Ordering::Relaxed));
            ((ceiling - external) / ceiling).clamp(0.5, 1.0)
        } else {
            1.0
        };
        // `ceiling` is clamped to 1..=100 on the way in, so the product bottoms
        // out at 0.005 (ceiling 1, fully yielded) and never reaches 0.
        ceiling / 100.0 * headroom
    }

    /// True when yielding and the load of other processes
    /// ([`external_utilization`](Self::external_utilization)) exceeds the
    /// ceiling. The cluster anneal of `quip-metal-otsuka` stops on this
    /// signal, as the CUDA governor's `should_throttle` does, so it never
    /// yields to its own load. The caller must report its GPU time through
    /// [`record_gpu_busy_us`](Self::record_gpu_busy_us).
    #[must_use]
    pub fn should_yield_to_others(&self) -> bool {
        self.knobs.yielding.load(Ordering::Relaxed)
            && self.knobs.external_util.load(Ordering::Relaxed)
                > self.knobs.ceiling.load(Ordering::Relaxed)
    }

    /// True when yielding and the last util sample exceeds the ceiling.
    pub fn should_throttle(&self) -> bool {
        self.knobs.yielding.load(Ordering::Relaxed)
            && self.knobs.last_util.load(Ordering::Relaxed)
                > self.knobs.ceiling.load(Ordering::Relaxed)
    }

    /// Request the poller to exit and join it.
    pub fn stop(&mut self) {
        self.knobs.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.thread().unpark();
            // A panicking poll thread is not fatal: the governor's whole
            // contract is to degrade to util 0 (see the module docs), which is
            // what a dead thread produces anyway — `last_util` simply stops
            // advancing and `should_throttle` goes quiet. But `poll_loop` has
            // no fallible step (atomic loads, a sleep, and a sensor read that
            // swallows every IOKit error), so a payload here means a real bug
            // in code that is supposed to be panic-free. Report it instead of
            // letting the miner silently mistake it for "no sensor available".
            if let Err(payload) = h.join() {
                // `Box<dyn Any>`'s own `Debug` only ever prints "Any", so dig
                // the message out of the two types `panic!` actually boxes.
                let msg = payload
                    .downcast_ref::<&'static str>()
                    .copied()
                    .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("<non-string panic payload>");
                tracing::warn!(
                    panic = msg,
                    "GPU util poll thread panicked; util stuck at 0"
                );
            }
        }
    }
}

impl Drop for UtilGovernor {
    fn drop(&mut self) {
        self.stop();
    }
}

// `device_index` is unused: IOKit's `IOAccelerator` matching walks every GPU
// service on the host (there is normally exactly one on Apple Silicon) and
// has no per-index selector analogous to NVML's `device_by_index`.
fn poll_loop(_device_index: u32, knobs: &Knobs) {
    let mut window_start = Instant::now();
    while !knobs.stop.load(Ordering::Relaxed) {
        // Sample unconditionally: `utilization()` feeds `Status.utilization`,
        // the miner's health report, which must be truthful whether or not
        // yielding is on. Gating the sample on `yielding` (as this once did)
        // made a fully-busy miner report 0% in the default configuration.
        // `yielding` still gates `should_throttle` — reporting load and acting
        // on it are separate concerns.
        let util = query_iokit_gpu_utilization();
        knobs.last_util.store(util, Ordering::Relaxed);

        // Convert the batch-reported GPU-busy time into our share of the window.
        // `swap` drains the accumulator so each window is independent; a batch
        // that reports between the swap and the store lands in the next window,
        // which is fine at these timescales.
        let now = Instant::now();
        let elapsed_us = now.duration_since(window_start).as_micros();
        window_start = now;
        let busy_us = u128::from(knobs.busy_us.swap(0, Ordering::Relaxed));
        let ours = self_util_pct(busy_us, elapsed_us);
        knobs.self_util.store(ours, Ordering::Relaxed);
        knobs
            .external_util
            .store(util.saturating_sub(ours), Ordering::Relaxed);
        // Yielding needs a fresh sample to act on: the throttle decides whether
        // to hold back the next dispatch, so a 2 s-stale reading would let a
        // whole batch through after pressure appeared, and hold back several
        // after it cleared. Idle-time reporting has no such deadline.
        let interval = if knobs.yielding.load(Ordering::Relaxed) {
            YIELDING_POLL
        } else {
            REPORTING_POLL
        };
        // stop() leaves a wake token even if it races with entering this wait.
        thread::park_timeout(interval);
    }
}

/// Our share of a poll window, as a percentage clamped to 0..=100.
///
/// Double-buffering keeps several command buffers in flight at once, so summed
/// device time legitimately exceeds the wall-clock window. Clamping (rather than
/// letting it through) matters because the caller subtracts this from the sensor
/// reading: an unclamped 300% would wrap the subtraction into a nonsense
/// "external" figure and collapse the dispatch budget to its floor.
fn self_util_pct(busy_us: u128, elapsed_us: u128) -> u32 {
    if elapsed_us == 0 {
        return 0;
    }
    u32::try_from(busy_us * 100 / elapsed_us)
        .unwrap_or(100)
        .min(100)
}

/// Sensor poll interval while yielding — fast enough that the throttle acts on
/// a current reading, slow enough that the IOKit walk stays negligible.
const YIELDING_POLL: Duration = Duration::from_millis(250);
/// Poll interval when only `Status.utilization` depends on the sample.
const REPORTING_POLL: Duration = Duration::from_secs(2);

/// Wrap a UTF-8 Rust string as a `CFStringRef`, or `None` on any failure.
///
/// Follows CoreFoundation's **Create Rule**: the returned reference is owned by
/// the caller (+1 retain), who must `CFRelease` it. Every call site in this
/// module does so immediately after the one `CFDictionaryGetValue` lookup it
/// was created for.
fn cfstr(s: &str) -> Option<CFStringRef> {
    let c = CString::new(s).ok()?;
    // SAFETY: `c` is a live NUL-terminated buffer that outlives this call, and
    // its contents are valid UTF-8 because `CString::new` was handed a `&str`,
    // matching the `kCFStringEncodingUTF8` we declare. A null allocator selects
    // the default allocator, which is the documented way to spell "no custom
    // allocator". CoreFoundation only reads the buffer — it copies the bytes
    // into the new CFString rather than borrowing them — so `c` is free to drop
    // at the end of this function.
    let r =
        unsafe { CFStringCreateWithCString(std::ptr::null(), c.as_ptr(), kCFStringEncodingUTF8) };
    if r.is_null() {
        None
    } else {
        Some(r)
    }
}

/// Walk every `IOAccelerator` service, handing each one's property dictionary
/// to `visit`.
///
/// This is the shared half of [`query_iokit_gpu_utilization`] and
/// [`gpu_core_count`]; they differ only in which property they pull out of the
/// dictionary and how they fold the results. Both get their "degrade quietly on
/// any failure" contract from this function returning early — a caller that
/// starts from a neutral accumulator and never sees `visit` run reports exactly
/// the same thing as one that found no matching service.
///
/// `visit` receives a **borrowed** `CFDictionaryRef` under the Get Rule: it is
/// live only for the duration of the call, and is released as soon as `visit`
/// returns. A callback must read what it needs and must not retain or stash the
/// pointer without taking its own reference.
fn for_each_accelerator_properties(mut visit: impl FnMut(CFDictionaryRef)) {
    let Ok(service_name) = CString::new("IOAccelerator") else {
        return;
    };
    // SAFETY: `service_name` is a live NUL-terminated buffer that outlives the
    // call, and `IOServiceMatching` only reads it. The returned dictionary
    // comes to us under the Create Rule (+1 owned), but we deliberately never
    // release it — see the ownership note on `IOServiceGetMatchingServices`.
    let matching = unsafe { IOServiceMatching(service_name.as_ptr()) };
    if matching.is_null() {
        return;
    }

    let mut iterator: io_iterator_t = 0;
    // SAFETY: `matching` is a non-null dictionary we own (null-checked above)
    // and `iterator` is a live, initialized out-param.
    //
    // This call *consumes* the `matching` reference: IOKitLib serializes the
    // dictionary and then unconditionally `CFRelease`s it, on the success and
    // the failure paths alike. Releasing `matching` ourselves — here or on the
    // early return below — would therefore be an over-release. The one path
    // that does not consume it is a null dictionary, which we already excluded.
    //
    // The iterator is the reference we *do* own; it is released at the end of
    // the walk. On a non-zero return the out-param is left at its `0`
    // initializer, so the early return below has nothing to clean up.
    let ret = unsafe {
        IOServiceGetMatchingServices(
            kIOMasterPortDefault,
            matching as CFDictionaryRef,
            &mut iterator,
        )
    };
    if ret != 0 {
        return;
    }

    loop {
        // SAFETY: `iterator` came from a successful `IOServiceGetMatchingServices`
        // and is still live. Each non-zero `io_object_t` is returned +1 owned
        // and is released below; 0 marks the end of the sequence.
        let service = unsafe { IOIteratorNext(iterator) };
        if service == 0 {
            break;
        }

        let mut props: CFMutableDictionaryRef = std::ptr::null_mut();
        // SAFETY: `service` is a live object from the iterator and `props` is a
        // live out-param pre-initialized to null; a null allocator selects the
        // default one.
        //
        // On success `props` is a +1 owned dictionary (Create Rule), released
        // after `visit`. A non-zero return never leaves an owned dictionary
        // behind: IOKitLib either returns before writing the out-param at all,
        // or writes it and then returns an error *precisely* when what it wrote
        // was null. So the `pret != 0 || props.is_null()` bail below cannot
        // strand an allocation.
        let pret =
            unsafe { IORegistryEntryCreateCFProperties(service, &mut props, std::ptr::null(), 0) };
        // SAFETY: balances the +1 from `IOIteratorNext`. Releasing the service
        // this early is sound because the property dictionary is a freshly
        // unserialized, independently-owned object rather than a view into the
        // registry entry, so it stays valid after its service goes away.
        unsafe { IOObjectRelease(service) };
        if pret != 0 || props.is_null() {
            continue;
        }

        visit(props as CFDictionaryRef);

        // SAFETY: balances the +1 from `IORegistryEntryCreateCFProperties`.
        // `visit` only borrows the dictionary (documented above), so this drops
        // the last reference.
        unsafe { CFRelease(props as CFTypeRef) };
    }

    // SAFETY: balances the iterator reference retained by
    // `IOServiceGetMatchingServices`. Only reachable when that call succeeded.
    unsafe { IOObjectRelease(iterator) };
}

/// GPU utilization percent (0-100) via IOKit, or 0 on any error.
///
/// Walks the `IOAccelerator` service(s), reading
/// `PerformanceStatistics -> "Device Utilization %"` from the IORegistry.
/// Never panics; a query failure (missing service, unsupported key, ...)
/// degrades to 0, matching the Python reference's `except Exception: return 0`.
fn query_iokit_gpu_utilization() -> u32 {
    let mut best: i64 = 0;
    for_each_accelerator_properties(|props| {
        // SAFETY: `props` is a live, borrowed property dictionary for the whole
        // of this callback, which is exactly `read_device_utilization`'s
        // precondition. It reads without taking ownership.
        if let Some(util) = unsafe { read_device_utilization(props) } {
            best = best.max(util);
        }
    });
    best.clamp(0, 100) as u32
}

/// Best-effort Apple GPU core count via IOKit, or `None` on any failure.
///
/// Reads the `"gpu-core-count"` integer property published on the
/// `IOAccelerator` service (Apple Silicon). Used only to size the streaming
/// command-buffer budget ([`crate::streaming::stream_width`]); callers fall
/// back to a conservative default when this returns `None`, so a miss costs
/// concurrency tuning, never correctness. Never panics — same "return nothing
/// on any error" contract as [`query_iokit_gpu_utilization`].
///
/// The core count is fixed hardware, and the registry query costs tens of
/// microseconds, so the first answer is kept for the life of the process.
pub fn gpu_core_count() -> Option<usize> {
    static CORES: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CORES.get_or_init(query_gpu_core_count)
}

fn query_gpu_core_count() -> Option<usize> {
    let mut cores: Option<usize> = None;
    for_each_accelerator_properties(|props| {
        // SAFETY: `props` is a live, borrowed property dictionary for the whole
        // of this callback, which is exactly `read_int_property`'s
        // precondition. It reads without taking ownership.
        if let Some(v) = unsafe { read_int_property(props, "gpu-core-count") } {
            if v > 0 {
                cores = Some(cores.map_or(v as usize, |c| c.max(v as usize)));
            }
        }
    });
    cores
}

/// Read a top-level signed-integer property from a service property dict,
/// or `None` if the key is absent / not a `CFNumber`.
///
/// # Safety
///
/// `props` must be a non-null `CFDictionaryRef` that stays live for the whole
/// call — the caller has to hold a reference to it, not merely have seen one.
/// Only borrowed under the Get Rule: this function never releases `props`, and
/// the value it looks up is likewise borrowed, so it must not outlive `props`
/// (it does not — the value is decoded into an owned `i64` before returning).
unsafe fn read_int_property(props: CFDictionaryRef, key: &str) -> Option<i64> {
    let k = cfstr(key)?;
    let val = CFDictionaryGetValue(props, k as *const c_void);
    CFRelease(k as CFTypeRef);
    if val.is_null() || CFGetTypeID(val) != CFNumberGetTypeID() {
        return None;
    }
    let mut out: i64 = 0;
    let ok = CFNumberGetValue(
        val as CFNumberRef,
        kCFNumberSInt64Type,
        &mut out as *mut i64 as *mut c_void,
    );
    ok.then_some(out)
}

/// Read `PerformanceStatistics -> "Device Utilization %"` from one service's
/// property dictionary, or `None` if either key is absent / not a number.
///
/// # Safety
///
/// `props` must be a non-null `CFDictionaryRef` that stays live for the whole
/// call — the caller has to hold a reference to it, not merely have seen one.
/// Only borrowed under the Get Rule: neither `props` nor the nested
/// `PerformanceStatistics` dictionary is released here, and the nested one is
/// only valid because its parent is held alive by the caller for the duration.
/// The dynamic type of every borrowed value is checked before use, so a driver
/// publishing an unexpected CF type yields `None` rather than undefined
/// behavior.
unsafe fn read_device_utilization(props: CFDictionaryRef) -> Option<i64> {
    let perf_key = cfstr("PerformanceStatistics")?;
    let perf_dict = CFDictionaryGetValue(props, perf_key as *const c_void);
    CFRelease(perf_key as CFTypeRef);
    // Guard the borrowed value's dynamic type before treating it as a
    // dictionary: a driver populating an unexpected CF type would otherwise
    // make the CFDictionaryGetValue call below undefined behavior.
    if perf_dict.is_null() || CFGetTypeID(perf_dict) != CFDictionaryGetTypeID() {
        return None;
    }

    let util_key = cfstr("Device Utilization %")?;
    let util_val = CFDictionaryGetValue(perf_dict as CFDictionaryRef, util_key as *const c_void);
    CFRelease(util_key as CFTypeRef);
    if util_val.is_null() || CFGetTypeID(util_val) != CFNumberGetTypeID() {
        return None;
    }

    let mut val: i64 = 0;
    let ok = CFNumberGetValue(
        util_val as CFNumberRef,
        kCFNumberSInt64Type,
        &mut val as *mut i64 as *mut c_void,
    );
    ok.then_some(val)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The foreign-load decision ignores our own share of the device, and it
    /// is off while not yielding.
    #[test]
    fn yielding_to_others_counts_only_foreign_load() {
        // Total 100 over the ceiling of 80, but 90 of it is ours.
        let own = UtilGovernor::with_sample_for_test(80, true, 100, 90);
        assert!(own.should_throttle());
        assert!(!own.should_yield_to_others());
        let foreign = UtilGovernor::with_sample_for_test(80, true, 100, 10);
        assert!(foreign.should_yield_to_others());
        let off = UtilGovernor::with_sample_for_test(80, false, 100, 10);
        assert!(!off.should_yield_to_others());
    }

    /// Live sensor read must never panic and must report a valid percentage.
    /// Exercises the real IOKit path (this crate is macOS-only).
    #[test]
    fn query_iokit_gpu_utilization_is_in_range() {
        let util = query_iokit_gpu_utilization();
        assert!(util <= 100, "util {util} out of 0..=100 range");
    }

    #[test]
    fn governor_without_yielding_never_throttles() {
        let mut gov = UtilGovernor::start(0, 100, false);
        assert!(!gov.should_throttle());
        gov.stop();
    }

    #[test]
    fn stop_wakes_the_reporting_poller() {
        let mut gov = UtilGovernor::start(0, 100, false);
        gov.record_gpu_busy_us(1);
        let deadline = Instant::now() + Duration::from_secs(5);
        // A drained report proves the poller has entered its sampling loop.
        while gov.knobs.busy_us.load(Ordering::Relaxed) != 0 {
            assert!(Instant::now() < deadline, "poller did not sample");
            thread::yield_now();
        }
        let started = Instant::now();
        gov.stop();
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    /// Utilization reporting must not depend on `yielding`: `Status.utilization`
    /// is the miner's health report, and a busy non-yielding miner reporting 0%
    /// is a lie the coordinator cannot detect. Only the range is asserted — the
    /// value is a live sensor reading, and a host with no sensor reports 0.
    #[test]
    fn utilization_is_reported_with_yielding_off() {
        let mut gov = UtilGovernor::start(0, 100, false);
        // One poll interval plus slack, so the first sample has landed.
        std::thread::sleep(REPORTING_POLL + Duration::from_millis(500));
        let util = gov.utilization();
        assert!(
            (0.0..=100.0).contains(&util),
            "util {util} out of 0..=100 range"
        );
        gov.stop();
    }

    /// The ceiling alone sizes the budget, yielding or not: `--utilization 80`
    /// means "aim for 80% of the device". The default 100 must cost nothing.
    #[test]
    fn budget_scale_follows_the_ceiling_without_yielding() {
        let mut gov = UtilGovernor::start(0, 100, false);
        assert!((gov.budget_scale() - 1.0).abs() < f64::EPSILON);
        gov.reconfigure(80, false);
        assert!((gov.budget_scale() - 0.8).abs() < 1e-9);
        gov.stop();
    }

    /// External load gives back ceiling one-for-one while yielding, but only
    /// down to half. Sustained contention must not stall mining outright.
    #[test]
    fn yielding_gives_back_ceiling_down_to_half() {
        let mut gov = UtilGovernor::start(0, 80, true);

        // No external load: full ceiling.
        gov.knobs.external_util.store(0, Ordering::Relaxed);
        assert!((gov.budget_scale() - 0.8).abs() < 1e-9);

        // 20 points external out of an 80 ceiling: 75% of the ceiling left.
        gov.knobs.external_util.store(20, Ordering::Relaxed);
        assert!((gov.budget_scale() - 0.8 * 0.75).abs() < 1e-9);

        // Saturating external load clamps at half the ceiling — 80% -> 40%,
        // exactly the floor, and it stays there no matter how high external is.
        gov.knobs.external_util.store(100, Ordering::Relaxed);
        assert!((gov.budget_scale() - 0.4).abs() < 1e-9);
        gov.stop();
    }

    /// Yielding is inert when off, whatever the external reading says.
    #[test]
    fn external_load_is_ignored_when_not_yielding() {
        let mut gov = UtilGovernor::start(0, 100, false);
        gov.knobs.external_util.store(100, Ordering::Relaxed);
        assert!((gov.budget_scale() - 1.0).abs() < f64::EPSILON);
        gov.stop();
    }

    /// Double-buffering keeps several command buffers in flight, so reported
    /// device time legitimately exceeds the wall-clock window. That must clamp
    /// to 100% rather than wrapping the `util - ours` subtraction into a huge
    /// bogus "external" reading.
    #[test]
    fn self_util_clamps_when_batches_overlap() {
        // Half a window busy.
        assert_eq!(self_util_pct(500, 1_000), 50);
        // Exactly saturated.
        assert_eq!(self_util_pct(1_000, 1_000), 100);
        // Three overlapping buffers: 300% of wall, clamped.
        assert_eq!(self_util_pct(3_000, 1_000), 100);
        // A zero-length window cannot produce a rate.
        assert_eq!(self_util_pct(1_000, 0), 0);
        // Idle.
        assert_eq!(self_util_pct(0, 1_000), 0);
    }

    /// End-to-end through the poll thread: a batch report must show up as
    /// non-zero self-utilization within one window.
    #[test]
    fn reported_busy_time_becomes_self_utilization() {
        let mut gov = UtilGovernor::start(0, 100, true);
        // Saturate several consecutive windows so the assertion does not race
        // the drain: each tick consumes what it finds and resets.
        let deadline = Instant::now() + YIELDING_POLL * 4;
        let mut seen: f32 = 0.0;
        while Instant::now() < deadline {
            gov.record_gpu_busy_us(50_000);
            std::thread::sleep(Duration::from_millis(25));
            seen = seen.max(gov.self_utilization());
        }
        assert!(
            seen > 0.0,
            "reported GPU time never became self utilization"
        );
        gov.stop();
    }
}
