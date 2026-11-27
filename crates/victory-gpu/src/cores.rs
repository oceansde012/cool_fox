//! Which processors our own threads run on, where a processor has two kinds
//! of cores. `cpu_sets` and `keep_on_fast_cores` are fxceless/victory-fork's
//! (`framelog.rs`, commit 0b0074e0, GPL-3.0-or-later, as this).

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

/// The system's processor sets of group 0: each one's id, its efficiency
/// class (higher is faster) and its logical processor's number.
fn cpu_sets() -> Vec<(u32, u8, u8)> {
    use windows_sys::Win32::System::SystemInformation::{GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut bytes = 0u32;
    // SAFETY: a size query, then the call into a buffer of that size.
    unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut bytes, GetCurrentProcess(), 0) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as above; the buffer holds `bytes` bytes, aligned for the records.
    let ok = unsafe { GetSystemCpuSetInformation(buffer.as_mut_ptr().cast(), bytes, &mut bytes, GetCurrentProcess(), 0) };
    let mut sets = Vec::new();
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_CPU_SET_INFORMATION>()) };
        if info.Size == 0 {
            break;
        }
        // SAFETY: the record's type is CpuSet (the only one there is).
        let set = unsafe { &info.Anonymous.CpuSet };
        if set.Group == 0 {
            sets.push((set.Id, set.EfficiencyClass, set.LogicalProcessorIndex));
        }
        at += info.Size as usize;
    }
    sets
}

/// The sets of the fastest class, if there are slower ones too.
fn fast_of(sets: &[(u32, u8, u8)]) -> Vec<u32> {
    let best = sets.iter().map(|c| c.1).max().unwrap_or(0);
    let fast: Vec<u32> = sets.iter().filter(|c| c.1 == best).map(|c| c.0).collect();
    if fast.len() == sets.len() { Vec::new() } else { fast }
}

/// Of the sets of the logical processors the title asked none of its threads
/// to run on (`asked`, a bit each): the fast ones (`Some(true)`), the slow
/// ones (`Some(false)`) or all. Empty if the title asked for nothing yet.
fn left_of(sets: &[(u32, u8, u8)], asked: u64, fast: Option<bool>) -> Vec<u32> {
    if asked == 0 {
        return Vec::new();
    }
    let best = sets.iter().map(|c| c.1).max().unwrap_or(0);
    sets.iter()
        .filter(|c| (c.2 >= 64 || asked >> c.2 & 1 == 0) && fast.is_none_or(|f| (c.1 == best) == f))
        .map(|c| c.0)
        .collect()
}

/// The logical processors of group 0 the title's threads were asked to run
/// on, all of its asks together ([`title_affinity`]).
static TITLE_ASKED: AtomicU64 = AtomicU64::new(0);

/// The title asked the system to run one of its threads on the logical
/// processors `mask` of group `group` (`victory-hle`'s
/// `SetThreadGroupAffinity`). It does so as on the console: its busiest
/// workers on the first eight, the rest on the six after them; on an
/// i9-12900K those are the first seven fast cores with both threads of each.
pub fn title_affinity(mask: usize, group: u16) {
    if group != 0 {
        return;
    }
    // Each mask once in the log, with how often it was asked so far.
    static SEEN: Mutex<Vec<(u64, u32)>> = Mutex::new(Vec::new());
    {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        match seen.iter_mut().find(|s| s.0 == mask as u64) {
            Some(s) => s.1 += 1,
            None => {
                seen.push((mask as u64, 1));
                tracing::info!(mask = format_args!("{mask:#x}"), "the title pins a thread to these processors, the first time");
            }
        }
    }
    // A mask of every processor pins nothing.
    let all = sets().0.iter().filter(|c| c.2 < 64).fold(0u64, |all, c| all | 1 << c.2);
    if mask as u64 & all == all {
        return;
    }
    let was = TITLE_ASKED.fetch_or(mask as u64, Relaxed);
    // Our threads that are kept off them move with what it asks.
    if was | mask as u64 != was && (apart() || lane_apart()) {
        place_again();
    }
}

/// The switches of where our threads run, each read from its variable (on
/// unless `0`) or changed while the title runs (`crate::ab`).
const SWITCHES: [&str; 2] = ["VICTORY_PLACE_APART", "VICTORY_PLACE_LANE"];

/// Each of [`SWITCHES`] as changed while the title runs: 0 as its variable
/// says, 1 on, 2 off.
static CHANGED: [AtomicU8; 2] = [const { AtomicU8::new(0) }; 2];

/// Whether switch `i` of [`SWITCHES`] is on.
fn switched(i: usize) -> bool {
    match CHANGED[i].load(Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    // Named in `VICTORY_AB`: off until its first phase on.
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick(SWITCHES[i]) {
        return on;
    }
    static ON: [OnceLock<bool>; 2] = [const { OnceLock::new() }; 2];
    *ON[i].get_or_init(|| std::env::var_os(SWITCHES[i]).is_none_or(|v| v != "0"))
}

/// Whether the threads that translate lists run on the slow cores the
/// title leaves (on by default; `VICTORY_PLACE_APART=0`: where they were).
///
/// Why: the title's threads run where it asks, as on the console. On an
/// i9-12900K its eight busy workers are on logical processors 0-7 (four
/// fast cores with both their threads), its main thread and ten others on
/// 8-12, one on 13. Ours were on the fast cores too, and the slow ones
/// stood idle. The eight threads that translate lists are the most work of
/// ours, and nothing waits for one of them alone.
///
/// Not all of ours: kept off every processor the title asks for (the line's
/// thread on 14-15, every other one on the slow cores), the frame was 5 ms
/// longer (split-pl3-apart). The helpers the line's thread and the sender
/// wait for (the heaps' records, refreshes spread over threads) were
/// slower there.
fn apart() -> bool {
    switched(0)
}

/// Whether the thread that prepares submissions runs on the fast cores the
/// title leaves (on by default; `VICTORY_PLACE_LANE=0`: on all fast ones):
/// 14-15 on an i9-12900K, where nothing of the title's runs.
fn lane_apart() -> bool {
    switched(1)
}

/// One of [`SWITCHES`] is `on` from now on (diagnostics, `crate::ab`): every
/// thread of ours is placed again, and what each kind of ours ran since the
/// last change is logged.
#[cfg(feature = "diagnostics")]
pub(crate) fn placing(switch: &str, on: bool) {
    let Some(i) = SWITCHES.iter().position(|s| *s == switch) else {
        return;
    };
    CHANGED[i].store(if on { 1 } else { 2 }, Relaxed);
    ran_line();
    place_again();
}

/// Diagnostics: the processor time of our threads that asked, by the name
/// they asked with, per second since the last line.
#[cfg(feature = "diagnostics")]
fn ran_line() {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::GetThreadTimes;
    static LAST: Mutex<Option<(std::time::Instant, std::collections::HashMap<usize, u64>)>> = Mutex::new(None);
    let ours = OURS.lock().unwrap_or_else(|e| e.into_inner());
    let mut now = std::collections::HashMap::new();
    for (handle, _, _) in ours.iter() {
        // SAFETY: zeroed FILETIMEs are valid; a thread of ours, opened with
        // the right to ask.
        let mut t: [FILETIME; 4] = unsafe { std::mem::zeroed() };
        let [a, b, c, d] = &mut t;
        unsafe { GetThreadTimes(*handle as *mut std::ffi::c_void, a, b, c, d) };
        let of = |f: &FILETIME| u64::from(f.dwHighDateTime) << 32 | u64::from(f.dwLowDateTime);
        now.insert(*handle, of(&t[2]) + of(&t[3]));
    }
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    // (Two switches changed at once: one line.)
    if last.as_ref().is_some_and(|(since, _)| since.elapsed() < std::time::Duration::from_secs(1)) {
        return;
    }
    if let Some((since, before)) = last.as_ref() {
        let seconds = since.elapsed().as_secs_f64().max(1e-3);
        let mut by: std::collections::BTreeMap<&str, (f64, u32)> = std::collections::BTreeMap::new();
        for (handle, _, what) in ours.iter() {
            let ran = now[handle].saturating_sub(before.get(handle).copied().unwrap_or(now[handle]));
            let e = by.entry(what.as_str()).or_default();
            *e = (e.0 + ran as f64 / 1e4 / seconds, e.1 + 1);
        }
        let text: Vec<String> = by.iter().map(|(what, (ms, n))| format!("{what} x{n} {ms:.0}")).collect();
        tracing::info!("our threads' processor time, ms a second, over {seconds:.1} s: {}", text.join(", "));
    }
    *last = Some((std::time::Instant::now(), now));
}

/// `VICTORY_FAST_CORES`: 0: no thread asks; 1: the threads a frame waits for
/// one by one; 2 (the default): the threads they share their work with too;
/// 3: those that work beside the rest as well (lists translated, descriptors
/// made): more threads by then than the fast cores run at a time, while the
/// slow ones stood idle.
fn stage() -> u32 {
    static STAGE: OnceLock<u32> = OnceLock::new();
    *STAGE.get_or_init(|| std::env::var("VICTORY_FAST_CORES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(2))
}

/// Asks the system to run the calling thread, which a frame waits for, on
/// the fast cores only, where the processor has two kinds. That fork's submit
/// thread was found on a slow one in 1,529 looks of 24,021. (A selection of
/// processors is a preference the scheduler keeps unless nothing of it is
/// free.) The title's threads are not asked for.
pub(crate) fn keep_on_fast_cores(what: &str) {
    ask(what, 1);
}

/// [`keep_on_fast_cores`] for one of the threads a submission's preparing or
/// a landing shares its work with.
pub(crate) fn keep_helper_on_fast_cores(what: &str) {
    ask(what, 2);
}

/// [`keep_on_fast_cores`] for one of many threads whose work is ahead of what
/// a frame waits for.
pub(crate) fn keep_many_on_fast_cores(what: &str) {
    ask(what, 3);
}

/// The sets, and the fast ones among them.
fn sets() -> &'static (Vec<(u32, u8, u8)>, Vec<u32>) {
    static SETS: OnceLock<(Vec<(u32, u8, u8)>, Vec<u32>)> = OnceLock::new();
    SETS.get_or_init(|| {
        let sets = cpu_sets();
        let fast = fast_of(&sets);
        (sets, fast)
    })
}

/// Where a thread that asked as `what` at stage `from` is to run, and what
/// that is called; `None`: wherever the system likes.
fn chosen(what: &str, from: u32) -> Option<(Vec<u32>, &'static str)> {
    let (sets, fast) = sets();
    let asked = TITLE_ASKED.load(Relaxed);
    let kept = match what {
        "victory-list" if apart() => Some((false, "the slow cores the title leaves (VICTORY_PLACE_APART)")),
        "victory-submit" if lane_apart() => Some((true, "the fast cores the title leaves (VICTORY_PLACE_LANE)")),
        _ => None,
    };
    if let Some((fast_ones, kind)) = kept
        && asked != 0
    {
        let left = left_of(sets, asked, Some(fast_ones));
        if !left.is_empty() {
            return Some((left, kind));
        }
        // The title leaves none of that kind: where it would have been.
        static SAID: [std::sync::Once; 2] = [const { std::sync::Once::new() }; 2];
        SAID[usize::from(fast_ones)].call_once(|| {
            let cores = if fast_ones { "fast" } else { "slow" };
            tracing::info!(what, title = format_args!("{asked:#x}"), "the title leaves no {cores} cores: the thread stays where it was ({kind})");
        });
    }
    (stage() >= from && !fast.is_empty()).then(|| (fast.clone(), "the fast cores (VICTORY_FAST_CORES=0: not)"))
}

/// Our threads that asked: a handle each, the stage and the name they asked
/// with.
static OURS: Mutex<Vec<(usize, u32, String)>> = Mutex::new(Vec::new());

fn ask(what: &str, from: u32) {
    use windows_sys::Win32::System::Threading::{
        GetCurrentThreadId, OpenThread, THREAD_QUERY_LIMITED_INFORMATION, THREAD_SET_LIMITED_INFORMATION,
    };
    // SAFETY: this thread of our own process; the handle is kept for good
    // (our threads that ask live as long as the process).
    let handle = unsafe { OpenThread(THREAD_SET_LIMITED_INFORMATION | THREAD_QUERY_LIMITED_INFORMATION, 0, GetCurrentThreadId()) };
    if handle.is_null() {
        return;
    }
    let mut ours = OURS.lock().unwrap_or_else(|e| e.into_inner());
    ours.push((handle as usize, from, what.to_string()));
    // (Under the lock: a placing again does not come between. Nothing to
    // choose: as before, nothing asked.)
    if let Some((sets, kind)) = chosen(what, from) {
        select(handle as usize, what, &sets, kind, true);
    }
}

/// Asks for `sets` (none: the system's choice) for the thread `handle`.
fn select(handle: usize, what: &str, sets: &[u32], kind: &str, say: bool) {
    use windows_sys::Win32::System::Threading::SetThreadSelectedCpuSets;
    let (ids, count) = if sets.is_empty() { (std::ptr::null(), 0) } else { (sets.as_ptr(), sets.len() as u32) };
    // SAFETY: a thread of ours, opened for this; a list of the system's own
    // set ids, or none.
    let ok = unsafe { SetThreadSelectedCpuSets(handle as *mut std::ffi::c_void, ids, count) };
    if ok == 0 && say {
        tracing::warn!(what, error = %std::io::Error::last_os_error(), "the thread could not be kept on {kind}");
    } else if ok != 0 && say {
        let title = TITLE_ASKED.load(Relaxed);
        tracing::info!(what, sets = sets.len(), of = self::sets().0.len(), title = format_args!("{title:#x}"), "the thread is kept on {kind}");
    }
}

/// Every thread of ours that asked is placed again, as [`chosen`] says now.
fn place_again() {
    let ours = OURS.lock().unwrap_or_else(|e| e.into_inner());
    for (handle, from, what) in ours.iter() {
        match chosen(what, *from) {
            Some((sets, kind)) => select(*handle, what, &sets, kind, false),
            None => select(*handle, what, &[], "the processors the system chooses", false),
        }
    }
    tracing::info!(
        threads = ours.len(),
        apart = apart(),
        lane = lane_apart(),
        title = format_args!("{:#x}", TITLE_ASKED.load(Relaxed)),
        "our threads are placed again (VICTORY_PLACE_*)"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fast_cores_are_those_of_the_best_class_if_there_is_a_worse_one() {
        assert_eq!(fast_of(&[(256, 1, 0), (257, 1, 1), (258, 0, 2)]), [256, 257]);
        assert!(fast_of(&[(256, 0, 0), (257, 0, 1)]).is_empty());
        assert!(fast_of(&[]).is_empty());
        assert!(!cpu_sets().is_empty());
        std::thread::spawn(|| keep_on_fast_cores("test")).join().unwrap();
    }

    /// The title on the first two fast processors of three: of the rest,
    /// the fast one, the slow one, or both.
    #[test]
    fn what_the_title_leaves_is_found_by_kind() {
        let four = [(256, 1, 0), (257, 1, 1), (258, 1, 2), (259, 0, 3)];
        assert_eq!(left_of(&four, 0b0011, Some(true)), [258]);
        assert_eq!(left_of(&four, 0b0011, Some(false)), [259]);
        assert_eq!(left_of(&four, 0b0011, None), [258, 259]);
        assert!(left_of(&four, 0, None).is_empty(), "the title asked for nothing yet");
        assert!(left_of(&four, 0b1111, None).is_empty(), "it left nothing");
        assert!(left_of(&four, 0b0111, Some(true)).is_empty(), "no fast one left");
    }
}
