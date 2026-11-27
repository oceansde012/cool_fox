//! Diagnostics: whether anything reads or writes the title memory whose
//! copy in the card's memory is not landed (`VICTORY_LAND_NOT_PROBE=1`, with
//! `VICTORY_LAND_NOT`). At each landing that leaves a piece on the card, the
//! piece's title memory is made no-access (if it is read-write, or read-only
//! as the guards leave a block). The first access to it comes here: the
//! protection it had is given back, the access is noted (address, read or
//! write, the instruction's module and offset, the thread) and said in the
//! log, and it goes on as it would have (a write to a guarded block then
//! faults again, for the guards).
//!
//! shortcut: a guard changed while its block is armed may be given back as
//! it was before (the probe's runs are for looking, not for playing).

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

/// Pieces that can be armed at once.
const SLOTS: usize = 4096;
/// Accesses noted between two log lines.
const HITS: usize = 64;

struct Slot {
    at: AtomicUsize,
    len: AtomicUsize,
    old: AtomicU32,
    armed: AtomicBool,
}

static ARMED: [Slot; SLOTS] = [const {
    Slot { at: AtomicUsize::new(0), len: AtomicUsize::new(0), old: AtomicU32::new(0), armed: AtomicBool::new(false) }
}; SLOTS];

/// Accesses: instruction, address, thread, write; and how many there were.
static HIT: [[AtomicUsize; 4]; HITS] = [const { [const { AtomicUsize::new(0) }; 4] }; HITS];
static HIT_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Pieces armed since the last log line.
static ARMINGS: AtomicUsize = AtomicUsize::new(0);

/// See the file.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAND_NOT_PROBE").is_some_and(|v| v == "1"))
}

/// The piece of title memory `[at, at + len)` (whole pages) is not landed:
/// made no-access until something touches it.
pub(crate) fn arm(at: usize, len: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualProtect, VirtualQuery,
    };
    install();
    let (at, end) = (at.next_multiple_of(4096), (at + len) / 4096 * 4096);
    if at >= end {
        return;
    }
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let queried = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } != 0;
    if !queried
        || info.State != MEM_COMMIT
        || !matches!(info.Protect, PAGE_READWRITE | PAGE_READONLY)
        || info.BaseAddress as usize + info.RegionSize < end
    {
        return;
    }
    // The piece's slot (it was armed before), or a free one.
    static CLAIM: Mutex<()> = Mutex::new(());
    let _claim = CLAIM.lock().unwrap_or_else(|e| e.into_inner());
    let slot = ARMED.iter().find(|s| s.at.load(Relaxed) == at).or_else(|| ARMED.iter().find(|s| s.at.load(Relaxed) == 0));
    let Some(slot) = slot else {
        return;
    };
    if slot.armed.load(Relaxed) {
        return;
    }
    slot.len.store(end - at, Relaxed);
    slot.old.store(info.Protect, Relaxed);
    slot.at.store(at, Relaxed);
    // Armed before the protection changes: a fault right after it finds it.
    slot.armed.store(true, Relaxed);
    let mut old = 0;
    // SAFETY: committed read-write memory of the title's, whole pages.
    if unsafe { VirtualProtect(at as *const _, end - at, PAGE_NOACCESS, &mut old) } == 0 {
        slot.armed.store(false, Relaxed);
        return;
    }
    ARMINGS.fetch_add(1, Relaxed);
}

fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
    };
    use windows_sys::Win32::System::Memory::VirtualProtect;
    use windows_sys::Win32::System::Threading::GetCurrentThreadId;
    unsafe extern "system" fn touched(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let address = r.ExceptionInformation[1];
        // (No allocation and no lock here: the thread may hold any.)
        let Some(slot) = ARMED.iter().find(|s| {
            let at = s.at.load(Relaxed);
            at != 0 && (at..at + s.len.load(Relaxed)).contains(&address)
        }) else {
            return EXCEPTION_CONTINUE_SEARCH;
        };
        if slot.armed.swap(false, Relaxed) {
            let mut old = 0;
            // SAFETY: the pages this module made no-access.
            unsafe { VirtualProtect(slot.at.load(Relaxed) as *const _, slot.len.load(Relaxed), slot.old.load(Relaxed), &mut old) };
            let n = HIT_COUNT.fetch_add(1, Relaxed);
            if let Some(hit) = HIT.get(n) {
                hit[0].store(r.ExceptionAddress as usize, Relaxed);
                hit[1].store(address, Relaxed);
                // SAFETY: a plain call.
                hit[2].store(unsafe { GetCurrentThreadId() } as usize, Relaxed);
                hit[3].store(r.ExceptionInformation[0], Relaxed);
            }
        }
        // (Given back by another thread meanwhile: tried again all the same.)
        EXCEPTION_CONTINUE_EXECUTION
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches its own state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(touched));
    });
}

/// An instruction's module and offset in it.
fn place(rip: usize) -> String {
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleFileNameW,
        GetModuleHandleExW,
    };
    let mut module = std::ptr::null_mut();
    // SAFETY: an address looked up, no reference taken.
    let found = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            rip as *const u16,
            &mut module,
        )
    } != 0;
    if !found {
        return format!("{rip:#x} (no module)");
    }
    let mut name = [0u16; 260];
    // SAFETY: a buffer of its length.
    let n = unsafe { GetModuleFileNameW(module, name.as_mut_ptr(), name.len() as u32) } as usize;
    let name = String::from_utf16_lossy(&name[..n]);
    let short = name.rsplit(['\\', '/']).next().unwrap_or(&name).to_owned();
    format!("{short}+{:#x}", rip - module as usize)
}

/// For the frame-rate line: pieces armed, and the accesses since.
pub(crate) fn log() {
    if !on() {
        return;
    }
    let armings = ARMINGS.swap(0, Relaxed);
    let hits = HIT_COUNT.swap(0, Relaxed);
    let armed = ARMED.iter().filter(|s| s.armed.load(Relaxed)).count();
    tracing::info!(
        "pieces not landed made no-access (VICTORY_LAND_NOT_PROBE): {armings} armed since the last line, {armed} armed now; \
         {hits} touched"
    );
    for hit in HIT.iter().take(hits.min(HITS)) {
        let [rip, address, thread, write] = [0, 1, 2, 3].map(|i| hit[i].load(Relaxed));
        tracing::warn!(
            address = format_args!("{address:#x}"),
            write = write == 1,
            thread,
            "title memory not landed was touched, by {}",
            place(rip)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_access_to_an_armed_piece_is_noted_and_goes_on() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        // SAFETY: fresh memory of ours, 64 KiB.
        let at = unsafe { VirtualAlloc(std::ptr::null(), 1 << 16, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(at, 0);
        // SAFETY: inside it.
        unsafe { ((at + 8192) as *mut u32).write_volatile(77) };
        let before = HIT_COUNT.load(Relaxed);
        arm(at, 1 << 16);
        assert!(ARMED.iter().any(|s| s.at.load(Relaxed) == at && s.armed.load(Relaxed)));
        // SAFETY: inside it; no-access now, given back by the handler.
        let read = unsafe { ((at + 8192) as *const u32).read_volatile() };
        assert_eq!(read, 77);
        assert_eq!(HIT_COUNT.load(Relaxed), before + 1);
        assert_eq!(HIT[before][1].load(Relaxed), at + 8192);
        // Given back: the next access is no fault.
        // SAFETY: as above.
        unsafe { ((at + 4) as *mut u32).write_volatile(1) };
        assert_eq!(HIT_COUNT.load(Relaxed), before + 1);
        // SAFETY: ours.
        unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) };
    }
}
