//! Diagnostics: where the title's bytes under flights' writes are
//! (`VICTORY_UNDER_WHO=1`). Per 64 KiB block of title memory, since the
//! start: the pages [`crate::shadow::Shadow::merge_under_writes`] put the
//! title's bytes into and their changed runs; how often a landing's plan
//! declared the block's pieces written, and how often the card's compare
//! found them changed; and the declarations of a megabyte or more
//! (`crate::host::wide_write`) over those blocks, by what declares them.
//! Every tenth line the busiest spans of blocks are logged.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::BLOCK;

/// Whether to look.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UNDER_WHO").is_some_and(|v| v == "1"))
}

#[derive(Default, Clone, Copy)]
struct Block {
    pages: u64,
    runs: u64,
    declared: u64,
    changed: u64,
    /// The copy: its title address and bytes.
    copy: (usize, usize),
}

#[derive(Default)]
struct All {
    blocks: HashMap<usize, Block>,
    /// Wide declarations over merged blocks: (from, to, what) and how often.
    wide: HashMap<(usize, usize, String), u64>,
    frames: f64,
    lines: u64,
}

static ALL: Mutex<Option<All>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut All) -> R) -> R {
    f(ALL.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(All::default))
}

/// The title's bytes went into the page at `page` (title memory) of the
/// copy at `copy`, `runs` changed runs of them.
pub(crate) fn merged(copy: (usize, usize), page: usize, runs: usize) {
    with(|all| {
        let b = all.blocks.entry(page / BLOCK).or_default();
        b.pages += 1;
        b.runs += runs as u64;
        b.copy = copy;
    });
}

/// A landing's plan declared title memory `[from, to)` written, and the
/// card's compare found it `changed` (or did not compare it).
pub(crate) fn declared(from: usize, to: usize, changed: bool) {
    if from >= to {
        return;
    }
    with(|all| {
        for block in from / BLOCK..=(to - 1) / BLOCK {
            if let Some(b) = all.blocks.get_mut(&block) {
                b.declared += 1;
                b.changed += u64::from(changed);
            }
        }
    });
}

/// At each counters' line: the wide declarations of the last `frames`
/// (from, to, what declares them: how often).
pub(crate) fn log(frames: f64, wide: &[((usize, usize), u64, String)]) {
    with(|all| {
        all.frames += frames;
        all.lines += 1;
        for ((from, to), n, what) in wide {
            if (from / BLOCK..to.div_ceil(BLOCK)).any(|b| all.blocks.contains_key(&b)) {
                *all.wide.entry((*from, *to, what.clone())).or_default() += n;
            }
        }
        if !all.lines.is_multiple_of(10) || all.frames == 0.0 {
            return;
        }
        // Spans of blocks next to one another in one copy.
        let mut keys: Vec<usize> = all.blocks.keys().copied().collect();
        keys.sort_unstable();
        let mut spans: Vec<(usize, usize, Block)> = Vec::new();
        for k in keys {
            let b = all.blocks[&k];
            match spans.last_mut() {
                Some((_, to, s)) if *to == k && s.copy == b.copy => {
                    *to = k + 1;
                    s.pages += b.pages;
                    s.runs += b.runs;
                    s.declared += b.declared;
                    s.changed += b.changed;
                }
                _ => spans.push((k, k + 1, b)),
            }
        }
        spans.sort_by_key(|s| std::cmp::Reverse(s.2.pages));
        let f = all.frames;
        // The busiest span block by block: where the card changed something,
        // and where it never did.
        if let Some(&(from, to, _)) = spans.first() {
            let blocks: Vec<(usize, Block)> = (from..to).filter_map(|k| all.blocks.get(&k).map(|b| (k, *b))).collect();
            let [never, ever] = [false, true].map(|changed| {
                blocks.iter().filter(|(_, b)| (b.changed > 0) == changed).fold((0u64, 0u64, 0u64), |(n, p, r), (_, b)| (n + 1, p + b.pages, r + b.runs))
            });
            let mut top: Vec<&(usize, Block)> = blocks.iter().collect();
            top.sort_by_key(|(_, b)| std::cmp::Reverse(b.pages));
            let top: Vec<String> = top
                .iter()
                .take(8)
                .map(|(k, b)| format!("{:#x}: {:.1} pages {:.0} runs, changed {}/{} declared", k * BLOCK, b.pages as f64 / f, b.runs as f64 / f, b.changed, b.declared))
                .collect();
            tracing::info!(
                "title bytes under flights' writes, the busiest span by block (VICTORY_UNDER_WHO): {} blocks the card never changed, \
                 {:.1} pages and {:.0} runs a frame in them; {} blocks it changed, {:.1} pages and {:.0} runs a frame; the busiest: {}",
                never.0,
                never.1 as f64 / f,
                never.2 as f64 / f,
                ever.0,
                ever.1 as f64 / f,
                ever.2 as f64 / f,
                top.join("; "),
            );
        }
        for (from, to, s) in spans.iter().take(6) {
            let (a, e) = (from * BLOCK, to * BLOCK);
            let mut over: Vec<(&(usize, usize, String), &u64)> =
                all.wide.iter().filter(|(k, _)| k.0 < e && k.1 > a).collect();
            over.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
            let over: Vec<String> =
                over.iter().take(6).map(|((x, y, what), n)| format!("{x:#x}+{:#x} by {what} {:.1}x", y - x, **n as f64 / f)).collect();
            tracing::info!(
                "title bytes under flights' writes, since the start (VICTORY_UNDER_WHO): {a:#x}+{:#x} of the copy {:#x}+{:#x}: \
                 {:.1} pages and {:.0} runs a frame; its pieces declared written {:.1} times a frame, found changed by the card {:.2} \
                 ({} in all); declared over it: {}",
                e - a,
                s.copy.0,
                s.copy.1,
                s.pages as f64 / f,
                s.runs as f64 / f,
                s.declared as f64 / f,
                s.changed as f64 / f,
                s.changed,
                over.join(", "),
            );
        }
    });
}
