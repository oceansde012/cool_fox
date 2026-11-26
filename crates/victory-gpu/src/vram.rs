//! Less driver paging once the process passes its video memory budget,
//! with nothing drawn differently: residency priorities for what frames
//! use every time, and a reservation of the card's memory. Both are off
//! by default (`VICTORY_RESIDENCY_PRIORITY`, `VICTORY_VRAM_RESERVE_MB`).
//!
//! Measured on base f4b48ad (RTX 5060 Ti 16 GB): the card's memory peaks at
//! 15.8 of 16 GB while the world streams in, and frames then take 0.25 to
//! 1.3 s with the GPU at 95 to 99 %: Windows' video memory manager moves
//! allocations to system memory and back, and without priorities it picks
//! among all of them alike, render targets drawn every frame included.
//! (Idea from Marvisak/VIctory #2222, not merged there; code of our own.)
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Once, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

const MIB: u64 = 1 << 20;
const RESOURCE_FLAG_ALLOW_RENDER_TARGET: u32 = 0x1;
const RESOURCE_FLAG_ALLOW_DEPTH_STENCIL: u32 = 0x2;
const RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS: u32 = 0x4;

/// Successful [`keep_resident`] calls (said in the reservation line and
/// once at the 1st, 1,000th and 10,000th).
static KEPT: AtomicU64 = AtomicU64::new(0);

/// Whether host allocations get residency priorities
/// (`VICTORY_RESIDENCY_PRIORITY=1`; off by default).
pub(crate) fn priorities_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RESIDENCY_PRIORITY").is_some_and(|v| v == "1"))
}

/// What a host allocation is to a frame, for the system's choice of what
/// to demote to system memory once the process passes its budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Residency {
    /// Render targets, depth buffers, textures and buffers shaders write
    /// (the GPU writes them every frame, and they hold the only copy of
    /// what it wrote), card copies of title buffers. `HIGH`: demoted after
    /// everything else of ours. Not `MAXIMUM`, which the system keeps even
    /// past the budget at the cost of other processes.
    Frame,
    /// Textures uploaded from title memory, streamed and each read by some
    /// frames. `NORMAL`, what every allocation starts with: no call. These
    /// go first, and sampling one from system memory is slower without
    /// stalling the frame the way a render target there does.
    Streamed,
}

impl Residency {
    /// A host texture's, from its resource flags: render target, depth
    /// stencil or unordered access, or none of them.
    pub(crate) fn of_texture(flags: u32) -> Residency {
        let written =
            RESOURCE_FLAG_ALLOW_RENDER_TARGET | RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if flags & written != 0 { Residency::Frame } else { Residency::Streamed }
    }

    fn priority(self) -> D3D12_RESIDENCY_PRIORITY {
        match self {
            Residency::Frame => D3D12_RESIDENCY_PRIORITY_HIGH,
            Residency::Streamed => D3D12_RESIDENCY_PRIORITY_NORMAL,
        }
    }
}

/// Sets `object`'s residency priority as `residency` says, if
/// [`priorities_on`]. Called where the object is made.
///
/// Only for committed resources and heaps; never for descriptor heaps or
/// placed resources: on an RTX 5070 Ti (driver 621.24) a priority on a
/// shader-visible descriptor heap or on a placed texture removed the device
/// at the next submission (Marvisak/VIctory #2222, #2226).
pub(crate) fn keep_resident(device: &ID3D12Device, object: &impl Interface, residency: Residency) {
    if !priorities_on() || residency.priority() == D3D12_RESIDENCY_PRIORITY_NORMAL {
        return;
    }
    let (Ok(device1), Ok(pageable)) = (device.cast::<ID3D12Device1>(), object.cast::<ID3D12Pageable>()) else {
        return;
    };
    let priority = residency.priority();
    // SAFETY: one live pageable object of the device's and its priority,
    // both on our stack for the call, which keeps neither.
    match unsafe { device1.SetResidencyPriority(1, &Some(pageable), &priority) } {
        Ok(()) => {
            let n = KEPT.fetch_add(1, Ordering::Relaxed) + 1;
            if matches!(n, 1 | 1_000 | 10_000) {
                tracing::info!("residency priorities: {n} host allocations set to be demoted last");
            }
        }
        Err(e) => {
            static ONCE: Once = Once::new();
            ONCE.call_once(|| tracing::info!(error = %e, "no residency priorities on this host"));
        }
    }
}

/// Reserves the card's memory as `VICTORY_VRAM_RESERVE_MB` says: unset or
/// 0 none (the default), `max` all the system offers
/// (`AvailableForReservation`), a number that many MiB at most.
///
/// A reservation tells the system the least this process needs, so the
/// budget it gives does not drop under it when other processes ask (the
/// budget was seen to fall by some 900 MB at the first world frame, and
/// everything over it is then paged).
pub(crate) fn reserve(adapter: &IDXGIAdapter3) {
    static SETTING: OnceLock<Option<String>> = OnceLock::new();
    let setting = SETTING.get_or_init(|| std::env::var("VICTORY_VRAM_RESERVE_MB").ok()).as_deref();
    if reservation_bytes(setting, u64::MAX) == 0 {
        return;
    }
    let query = || {
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: a plain query that fills our structure.
        unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }.map(|()| info)
    };
    let before = match query() {
        Ok(info) => info,
        Err(e) => {
            tracing::info!(error = %e, "no video memory reservation: the query failed");
            return;
        }
    };
    let asked = reservation_bytes(setting, before.AvailableForReservation);
    if asked == 0 {
        return;
    }
    // SAFETY: a plain call on a live adapter.
    if let Err(e) = unsafe { adapter.SetVideoMemoryReservation(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, asked) } {
        tracing::info!(error = %e, asked_mb = asked / MIB, "no video memory reservation");
        return;
    }
    if let Ok(after) = query() {
        tracing::info!(
            "video memory reserved ({:?}): asked {} MB of {} MB offered; the card's budget {} MB, in use {} MB, reserved {} MB",
            setting.unwrap_or_default(),
            asked / MIB,
            before.AvailableForReservation / MIB,
            after.Budget / MIB,
            after.CurrentUsage / MIB,
            after.CurrentReservation / MIB,
        );
    }
}

/// The bytes [`reserve`] asks for, from the setting and what the system
/// offers: trimmed, `max` in any case all of `available`, a number that
/// many MiB but not more than `available`, anything else none.
pub(crate) fn reservation_bytes(setting: Option<&str>, available: u64) -> u64 {
    let Some(setting) = setting.map(str::trim) else {
        return 0;
    };
    if setting.eq_ignore_ascii_case("max") {
        return available;
    }
    setting.parse::<u64>().map_or(0, |mb| mb.saturating_mul(MIB).min(available))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_flags_map_to_residency() {
        assert_eq!(Residency::of_texture(0), Residency::Streamed);
        assert_eq!(Residency::of_texture(0x1), Residency::Frame);
        assert_eq!(Residency::of_texture(0x2), Residency::Frame);
        assert_eq!(Residency::of_texture(0x4), Residency::Frame);
        assert_eq!(Residency::of_texture(0x8), Residency::Streamed);
        assert_eq!(Residency::of_texture(0x1 | 0x8), Residency::Frame);
    }

    #[test]
    fn residency_priority_values() {
        assert_eq!(Residency::Frame.priority(), D3D12_RESIDENCY_PRIORITY_HIGH);
        assert_eq!(Residency::Streamed.priority(), D3D12_RESIDENCY_PRIORITY_NORMAL);
    }

    #[test]
    fn reservation_bytes_parses_settings() {
        assert_eq!(reservation_bytes(None, 10 * MIB), 0);
        assert_eq!(reservation_bytes(Some(""), 10 * MIB), 0);
        assert_eq!(reservation_bytes(Some("0"), 10 * MIB), 0);
        assert_eq!(reservation_bytes(Some("abc"), 10 * MIB), 0);
        assert_eq!(reservation_bytes(Some("512"), 2048 * MIB), 512 * MIB);
        assert_eq!(reservation_bytes(Some("512"), 100 * MIB), 100 * MIB);
        assert_eq!(reservation_bytes(Some("MAX"), 123 * MIB), 123 * MIB);
        assert_eq!(reservation_bytes(Some(" max "), 123 * MIB), 123 * MIB);
        assert_eq!(reservation_bytes(Some("18446744073709551615"), 2048 * MIB), 2048 * MIB);
    }
}
