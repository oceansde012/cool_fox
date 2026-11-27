//! Compares that read only the pieces likely to have changed
//! (`VICTORY_CARD_COMPARE_HOT=<D>`; off by default).
//!
//! After each submission the card compares every 64 KiB piece of every range
//! its shaders were declared to write (`CARD_COMPARE`). Walking in the prison
//! that is 7,600-13,400 pieces a frame (500-880 MB read) of which 90-95 % are
//! found unchanged, and the same few pieces are found changed frame after
//! frame. With the switch a submission's compare reads only the declared
//! pieces found changed lately ("hot": [`hot_span`] submissions); the others
//! are "owed": their stored hash is still the one from before this
//! submission's shaders ran, so a later compare of them still finds what the
//! shaders wrote, however many submissions wrote it meanwhile.
//!
//! What may not happen to an owed piece, and what this book does about it:
//!
//! - **its stored hash replaced before a compare has looked at it**: the hash
//!   passes after ranges are sent up (mode 0) would take the shaders' bytes
//!   into the hash, and no compare would ever flag them. A submission whose
//!   sends hash an owed piece again compares it first, in a list of its own
//!   ahead of the sends ("a cover"). The same for an owed piece that the
//!   submission's own compare would read (it stores its hash too: what it
//!   flags lands with this submission, not with the one that owes it).
//! - **the title told before what the shaders wrote is in title memory**:
//!   fences and bottom-of-pipe values are told when a flight lands, and the
//!   title may then read or rewrite those bytes. A flight that owes pieces
//!   does not land before a cover has compared all of them and that cover has
//!   run on the card ([`Book::hold`]); the cover's results land with it. Every
//!   point that waits for flights to land (a synchronous submission, a read of
//!   title memory a flight may write, a blocking fence wait, the lists in the
//!   air, the sweep) waits for that too: nothing changes for them.
//! - **owed for ever**: a piece owed `D` submissions is compared by the cover
//!   of the `D`-th one after; and the thread that lands flights compares every
//!   owed piece itself when the card has run everything in the line or a
//!   thread waits for landings ([`Book::take_all`]), so a title that stops
//!   submitting still has every flight land.
//!
//! The book is pure: the host keeps it under a lock, records the covers it
//! asks for and tells it what compares found.

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// `VICTORY_CARD_COMPARE_HOT=<D>`: the most submissions after its own that a
/// piece a submission skipped waits for its compare; unset or 0, every
/// declared piece is compared after every submission, as before. The flight
/// that owes it lands no earlier than that compare has run, so its fences are
/// told up to `D` submissions later (a frame is some 30).
pub(super) fn deadline() -> Option<u64> {
    static D: OnceLock<Option<u64>> = OnceLock::new();
    *D.get_or_init(|| {
        std::env::var("VICTORY_CARD_COMPARE_HOT").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&d| d > 0)
    })
}

/// `VICTORY_CARD_COMPARE_HOT_SPAN` (90, about three frames): for how many
/// submissions after a compare found a piece changed the piece is compared
/// after every submission that declares it.
pub(super) fn hot_span() -> u64 {
    static SPAN: OnceLock<u64> = OnceLock::new();
    *SPAN.get_or_init(|| {
        std::env::var("VICTORY_CARD_COMPARE_HOT_SPAN").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(90)
    })
}

/// Since the last frame-rate line ([`line`]): declared pieces compared as
/// hot, declared pieces left for later, owed pieces compared by covers (as
/// due, ahead of a send, ahead of the submission's own compare, by the thread
/// that lands flights), owed pieces found changed by covers, the most
/// submissions a changed one waited, the lists the landing thread made for
/// covers, the times it held a flight back for covers, microseconds it
/// waited for their compares.
pub(super) static STATS: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];

pub(super) mod stat {
    pub(crate) const HOT: usize = 0;
    pub(crate) const LEFT: usize = 1;
    pub(crate) const DUE: usize = 2;
    pub(crate) const SENT: usize = 3;
    pub(crate) const AHEAD: usize = 4;
    pub(crate) const FORCED: usize = 5;
    pub(crate) const FOUND: usize = 6;
    /// A maximum, not a sum.
    pub(crate) const WAITED: usize = 7;
    pub(crate) const FORCED_LISTS: usize = 8;
    pub(crate) const HELD: usize = 9;
    pub(crate) const HELD_US: usize = 10;
    /// Owed pieces when the line was written (a level, not a sum).
    pub(crate) const OWED: usize = 11;
}

pub(super) fn count(what: usize, n: u64) {
    STATS[what].fetch_add(n, Ordering::Relaxed);
}

/// The frame-rate line's part ([`STATS`]), if the switch is on.
pub(super) fn line(frames: f64) {
    let Some(d) = deadline() else {
        return;
    };
    let s: Vec<u64> = STATS
        .iter()
        .enumerate()
        .map(|(i, v)| if i == stat::OWED { v.load(Ordering::Relaxed) } else { v.swap(0, Ordering::Relaxed) })
        .collect();
    let f = |i: usize| s[i] as f64 / frames;
    tracing::info!(
        "compares of hot pieces only (VICTORY_CARD_COMPARE_HOT={d}, hot for {} submissions) per frame: {:.0} declared pieces \
         compared as hot, {:.0} left for later; {:.0} owed pieces compared later ({:.0} due, {:.0} ahead of a send, {:.0} ahead of \
         their submission's own compare, {:.0} by the landing thread in {:.1} lists of its own), {:.1} of them found changed, \
         the longest a changed one waited {} submissions; the landing thread held a flight back for them {:.1} times, \
         {:.2} ms waiting for their compares; \
         {} pieces owed now",
        hot_span(),
        f(stat::HOT),
        f(stat::LEFT),
        f(stat::DUE) + f(stat::SENT) + f(stat::AHEAD) + f(stat::FORCED),
        f(stat::DUE),
        f(stat::SENT),
        f(stat::AHEAD),
        f(stat::FORCED),
        f(stat::FORCED_LISTS),
        f(stat::FOUND),
        s[stat::WAITED],
        f(stat::HELD),
        f(stat::HELD_US) / 1000.0,
        s[stat::OWED],
    );
}

/// A piece no compare has looked at since a submission that may have written
/// it: the first such submission (its flight may not land before the piece
/// is compared), and the offsets of the copy its declared ranges took of the
/// piece (what its landing merges).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Owed {
    pub(super) first: u64,
    pub(super) lo: usize,
    pub(super) hi: usize,
}

/// An owed piece taken for a cover: the copy's number, its handle, the
/// piece, and what was owed.
pub(super) type Taken<S> = (u64, S, u32, Owed);

struct Copy<S> {
    handle: S,
    /// Piece, and the submission whose compare last found it changed.
    changed: HashMap<u32, u64>,
    owed: BTreeMap<u32, Owed>,
}

struct Cover<C> {
    /// Done when the cover fence has reached it.
    value: u64,
    /// The first submission that owed one of its pieces.
    oldest: u64,
    payload: C,
}

/// One declared range of a copy in a submission, piece by piece: the piece
/// and the offsets of the copy the range takes of it.
pub(super) struct Span<S> {
    pub(super) copy: u64,
    pub(super) handle: S,
    pub(super) pieces: Vec<(u32, usize, usize)>,
}

/// What [`Book::plan`] decided for a submission.
pub(super) struct Planned<S> {
    /// The owed pieces to compare ahead of its sends (a cover).
    pub(super) covers: Vec<Taken<S>>,
    /// Per span given, per piece: whether its own compare reads it.
    pub(super) chosen: Vec<Vec<bool>>,
}

/// Why a flight may not land yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Hold {
    Free,
    /// It, or one ahead of it, owes a piece no cover has taken.
    Uncovered,
    /// A cover of a piece it owes has not run: the cover fence's value.
    Cover(u64),
}

pub(super) struct Book<S, C> {
    copies: HashMap<u64, Copy<S>>,
    /// By the first submission that owes them: pieces owed and not taken.
    uncovered: BTreeMap<u64, u32>,
    covers: Vec<Cover<C>>,
    plans: u64,
}

impl<S: Clone, C> Default for Book<S, C> {
    fn default() -> Self {
        Book { copies: HashMap::new(), uncovered: BTreeMap::new(), covers: Vec::new(), plans: 0 }
    }
}

impl<S: Clone, C> Book<S, C> {
    /// Submission `number`'s compares. `spans`: its declared ranges of copies
    /// it compares (`flown`: sent ahead, its compare stores the hashes it
    /// makes; else it reads every piece, as it always did). `rehashed`: the
    /// pieces (copy, piece) its sends hash again.
    ///
    /// In this order, because each step changes what the next may do: which
    /// declared pieces are hot; which owed pieces a cover takes now (due, or
    /// to be hashed again by the sends, or read by the submission's own
    /// compare, which stores its hash); and the declared pieces not read
    /// become owed (by this submission, if no earlier one owes them still).
    pub(super) fn plan(
        &mut self,
        number: u64,
        deadline: u64,
        span: u64,
        spans: &[Span<S>],
        rehashed: &dyn Fn(u64, u32) -> bool,
        flown: bool,
    ) -> Planned<S> {
        self.plans += 1;
        if self.plans.is_multiple_of(1024) {
            self.prune(number, span);
        }
        let chosen: Vec<Vec<bool>> = spans
            .iter()
            .map(|s| {
                let copy = self.copies.get(&s.copy);
                s.pieces
                    .iter()
                    .map(|&(p, ..)| !flown || copy.and_then(|c| c.changed.get(&p)).is_some_and(|&n| n + span >= number))
                    .collect()
            })
            .collect();
        let read: std::collections::HashSet<(u64, u32)> = spans
            .iter()
            .zip(&chosen)
            .flat_map(|(s, c)| s.pieces.iter().zip(c).filter(|(_, c)| **c).map(|(p, _)| (s.copy, p.0)))
            .collect();
        let mut covers = Vec::new();
        for (&copy, state) in &mut self.copies {
            let taken: Vec<u32> = state
                .owed
                .iter()
                .filter(|(p, o)| {
                    let why = if o.first + deadline <= number {
                        Some(stat::DUE)
                    } else if rehashed(copy, **p) {
                        Some(stat::SENT)
                    } else if read.contains(&(copy, **p)) {
                        Some(stat::AHEAD)
                    } else {
                        None
                    };
                    why.inspect(|&w| count(w, 1)).is_some()
                })
                .map(|(p, _)| *p)
                .collect();
            for p in taken {
                let o = state.owed.remove(&p).expect("taken from the map");
                uncount(&mut self.uncovered, o.first);
                covers.push((copy, state.handle.clone(), p, o));
            }
        }
        for (s, c) in spans.iter().zip(&chosen) {
            let state = self.copies.entry(s.copy).or_insert_with(|| Copy {
                handle: s.handle.clone(),
                changed: HashMap::new(),
                owed: BTreeMap::new(),
            });
            state.handle = s.handle.clone();
            for (&(p, lo, hi), &chosen) in s.pieces.iter().zip(c) {
                if chosen {
                    count(stat::HOT, 1);
                    continue;
                }
                count(stat::LEFT, 1);
                match state.owed.get_mut(&p) {
                    Some(o) => (o.lo, o.hi) = (o.lo.min(lo), o.hi.max(hi)),
                    None => {
                        state.owed.insert(p, Owed { first: number, lo, hi });
                        *self.uncovered.entry(number).or_insert(0) += 1;
                    }
                }
            }
        }
        STATS[stat::OWED].store(self.owed() as u64, Ordering::Relaxed);
        Planned { covers, chosen }
    }

    /// Every owed piece, taken for a cover the thread that lands flights
    /// makes.
    pub(super) fn take_all(&mut self) -> Vec<Taken<S>> {
        let mut out = Vec::new();
        for (&copy, state) in &mut self.copies {
            for (p, o) in std::mem::take(&mut state.owed) {
                out.push((copy, state.handle.clone(), p, o));
            }
        }
        self.uncovered.clear();
        count(stat::FORCED, out.len() as u64);
        STATS[stat::OWED].store(0, Ordering::Relaxed);
        out
    }

    /// A cover of `taken` is on the queue, done when the cover fence reaches
    /// `value`; `payload` lands with the first flight that owed one of them.
    pub(super) fn add_cover(&mut self, value: u64, taken: &[Taken<S>], payload: C) {
        let oldest = taken.iter().map(|t| t.3.first).min().unwrap_or(0);
        self.covers.push(Cover { value, oldest, payload });
    }

    /// A compare of submission `number` found the piece changed.
    pub(super) fn found(&mut self, copy: u64, piece: u32, number: u64) {
        if let Some(state) = self.copies.get_mut(&copy) {
            let at = state.changed.entry(piece).or_insert(number);
            *at = (*at).max(number);
        }
    }

    /// Whether flight `flight` may land, the cover fence having reached
    /// `completed`. (Flights land in order: one ahead of it that owes is
    /// held, and so it is.)
    pub(super) fn hold(&self, flight: u64, completed: u64) -> Hold {
        if self.uncovered.range(..=flight).next().is_some() {
            return Hold::Uncovered;
        }
        match self.covers.iter().filter(|c| c.oldest <= flight && c.value > completed).map(|c| c.value).max() {
            Some(value) => Hold::Cover(value),
            None => Hold::Free,
        }
    }

    /// The covers that land with flights up to `last` (their first owing
    /// flight is one of them), done by `completed`.
    pub(super) fn take_landed(&mut self, last: u64, completed: u64) -> Vec<C> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.covers.len() {
            if self.covers[i].oldest <= last && self.covers[i].value <= completed {
                out.push(self.covers.swap_remove(i).payload);
            } else {
                i += 1;
            }
        }
        out
    }

    /// Pieces owed and not taken.
    pub(super) fn owed(&self) -> usize {
        self.uncovered.values().map(|&n| n as usize).sum()
    }

    /// Copies with nothing owed and nothing found lately go (their numbers
    /// are never given again; the book would only grow).
    fn prune(&mut self, number: u64, span: u64) {
        self.copies.retain(|_, c| {
            c.changed.retain(|_, n| *n + span >= number);
            !c.changed.is_empty() || !c.owed.is_empty()
        });
    }
}

fn uncount(uncovered: &mut BTreeMap<u64, u32>, first: u64) {
    if let Some(n) = uncovered.get_mut(&first) {
        *n -= 1;
        if *n == 0 {
            uncovered.remove(&first);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Book, Hold, Span};

    fn span(copy: u64, pieces: std::ops::Range<u32>) -> Span<()> {
        Span { copy, handle: (), pieces: pieces.map(|p| (p, p as usize * 65536, (p as usize + 1) * 65536)).collect() }
    }

    /// A piece found changed is read after every submission that declares
    /// it while it is hot, the others are owed and their flight is held
    /// until a cover took them and ran.
    #[test]
    fn hot_pieces_are_read_and_cold_ones_owed_until_covered() {
        let mut book: Book<(), &str> = Book::default();
        let none = |_: u64, _: u32| false;
        // Submission 1: nothing known, everything owed.
        let p = book.plan(1, 4, 10, &[span(7, 0..4)], &none, true);
        assert!(p.covers.is_empty());
        assert_eq!(p.chosen, vec![vec![false; 4]]);
        assert_eq!(book.owed(), 4);
        assert_eq!(book.hold(1, 0), Hold::Uncovered);
        // Submission 2 declares nothing; piece 2 of submission 1's range is
        // hashed again by its sends: taken first.
        let p = book.plan(2, 4, 10, &[], &|c, p| c == 7 && p == 2, true);
        assert_eq!(p.covers.iter().map(|t| t.2).collect::<Vec<_>>(), vec![2]);
        book.add_cover(1, &p.covers, "two");
        assert_eq!(book.hold(1, 0), Hold::Uncovered, "three pieces still owed");
        // Submission 5 is submission 1's deadline: the rest are taken.
        let p = book.plan(5, 4, 10, &[], &none, true);
        let mut taken: Vec<u32> = p.covers.iter().map(|t| t.2).collect();
        taken.sort_unstable();
        assert_eq!(taken, vec![0, 1, 3]);
        book.add_cover(2, &p.covers, "five");
        assert_eq!(book.owed(), 0);
        assert_eq!(book.hold(1, 0), Hold::Cover(2));
        assert_eq!(book.hold(1, 1), Hold::Cover(2));
        assert_eq!(book.hold(1, 2), Hold::Free);
        assert!(book.take_landed(1, 1).contains(&"two"));
        assert_eq!(book.take_landed(1, 2), vec!["five"]);
        // The cover found piece 3 changed: hot for ten submissions.
        book.found(7, 3, 5);
        let p = book.plan(6, 4, 10, &[span(7, 2..5)], &none, true);
        assert_eq!(p.chosen, vec![vec![false, true, false]]);
        let p = book.plan(15, 4, 10, &[span(7, 3..4)], &none, true);
        assert!(p.chosen[0][0], "still hot ten submissions after");
        // Pieces 2 and 4, owed by 6, were due at 10 = 6 + 4.
        let mut taken: Vec<(u32, u64)> = p.covers.iter().map(|t| (t.2, t.3.first)).collect();
        taken.sort_unstable();
        assert_eq!(taken, vec![(2, 6), (4, 6)]);
        let p = book.plan(16, 4, 10, &[span(7, 3..4)], &none, true);
        assert!(!p.chosen[0][0], "cold eleven after");
    }

    /// An owed piece that turns hot (a landing found it changed after it was
    /// skipped) is taken by a cover before the submission's own compare
    /// stores its hash: else that compare would flag what the owing flight
    /// wrote, for its own landing.
    #[test]
    fn an_owed_piece_read_by_a_later_compare_is_covered_first() {
        let mut book: Book<(), ()> = Book::default();
        let none = |_: u64, _: u32| false;
        book.plan(1, 100, 10, &[span(1, 0..2)], &none, true);
        book.found(1, 1, 0);
        let p = book.plan(2, 100, 10, &[span(1, 0..2)], &none, true);
        assert_eq!(p.chosen, vec![vec![false, true]]);
        assert_eq!(p.covers.iter().map(|t| (t.2, t.3.first)).collect::<Vec<_>>(), vec![(1, 1)]);
        // Piece 0 stays owed by submission 1 (not 2), so 1 is held.
        assert_eq!(book.hold(1, u64::MAX), Hold::Uncovered);
        assert_eq!(book.take_all().iter().map(|t| (t.2, t.3.first)).collect::<Vec<_>>(), vec![(0, 1)]);
        assert_eq!(book.hold(2, 0), Hold::Free);
    }

    /// A submission that is waited for reads every declared piece.
    #[test]
    fn a_submission_waited_for_reads_everything() {
        let mut book: Book<(), ()> = Book::default();
        let p = book.plan(1, 4, 10, &[span(1, 0..3)], &|_, _| false, false);
        assert_eq!(p.chosen, vec![vec![true; 3]]);
        assert_eq!(book.owed(), 0);
    }

    /// A model of the card, the queue and the landings, with the book doing
    /// what `Host::card_before` and the landing thread do: per piece the card
    /// holds a version (a shader write makes a new one), the stored hash is
    /// the version a compare or a hash pass last saw, and title memory holds
    /// the newest version a landing fetched. Submissions declare and write
    /// pieces, their sends hash pieces again, the card runs behind them and
    /// the landing thread lands what the book lets it, covering everything
    /// when the card stands idle. Whenever a flight lands (the title is told
    /// it is done), title memory must hold every write of it and of the
    /// flights before it.
    #[test]
    fn no_write_reaches_the_title_late_in_a_model_of_the_relay() {
        const PIECES: u32 = 12;
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rand = move |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for deadline in [1u64, 2, 5, 30] {
            let mut book: Book<(), Vec<u32>> = Book::default();
            let mut card = [0u64; PIECES as usize];
            let mut stored = [0u64; PIECES as usize];
            let mut title = [0u64; PIECES as usize];
            let mut version = 0u64;
            // Flights in the line: number, own flagged pieces, writes.
            let mut line: std::collections::VecDeque<(u64, Vec<u32>, Vec<(u32, u64)>)> = Default::default();
            let mut cover_value = 0u64;
            // The card runs in order; `ran` is the last submission run and
            // covers are done as soon as they are put on the queue (they run
            // ahead of their submission's lists, behind earlier ones: the
            // model applies every effect in queue order at once).
            let mut landings = 0;
            let compare = |pieces: &[u32], card: &[u64], stored: &mut [u64]| -> Vec<u32> {
                let mut flagged = Vec::new();
                for &p in pieces {
                    if card[p as usize] != stored[p as usize] {
                        flagged.push(p);
                    }
                    stored[p as usize] = card[p as usize];
                }
                flagged
            };
            for number in 1..=600u64 {
                // Declared: one or two runs of pieces.
                let mut declared: Vec<u32> = Vec::new();
                for _ in 0..1 + rand(2) {
                    let from = rand(u64::from(PIECES)) as u32;
                    let to = (from + 1 + rand(5) as u32).min(PIECES);
                    declared.extend(from..to);
                }
                declared.sort_unstable();
                declared.dedup();
                let sent: Vec<u32> = (0..rand(3)).map(|_| rand(u64::from(PIECES)) as u32).collect();
                let spans = [Span { copy: 1, handle: (), pieces: declared.iter().map(|&p| (p, 0, 1)).collect() }];
                let planned = book.plan(number, deadline, 6, &spans, &|_, p| sent.contains(&p), true);
                // The cover, ahead of the sends.
                let covered: Vec<u32> = planned.covers.iter().map(|t| t.2).collect();
                let flagged = compare(&covered, &card, &mut stored);
                cover_value += 1;
                book.add_cover(cover_value, &planned.covers, flagged);
                // The sends hash pieces again.
                for &p in &sent {
                    stored[p as usize] = card[p as usize];
                }
                // The lists write some declared pieces.
                let mut writes = Vec::new();
                for &p in &declared {
                    if rand(3) == 0 {
                        version += 1;
                        card[p as usize] = version;
                        writes.push((p, version));
                    }
                }
                // The submission's own compare reads the chosen pieces.
                let chosen: Vec<u32> = declared.iter().zip(&planned.chosen[0]).filter(|(_, c)| **c).map(|(p, _)| *p).collect();
                let own = compare(&chosen, &card, &mut stored);
                line.push_back((number, own, writes));
                // The landing thread, now and then; with the card idle (a
                // third of the time) it covers what is still owed.
                if rand(2) == 0 {
                    continue;
                }
                let idle = rand(3) == 0;
                loop {
                    let Some(front) = line.front() else {
                        break;
                    };
                    match book.hold(front.0, cover_value) {
                        Hold::Free => {}
                        Hold::Cover(_) => unreachable!("covers are done at once here"),
                        Hold::Uncovered if idle => {
                            let taken = book.take_all();
                            let covered: Vec<u32> = taken.iter().map(|t| t.2).collect();
                            let flagged = compare(&covered, &card, &mut stored);
                            cover_value += 1;
                            book.add_cover(cover_value, &taken, flagged);
                            continue;
                        }
                        Hold::Uncovered => break,
                    }
                    let (flight, own, writes) = line.pop_front().unwrap();
                    // Fetched now: whatever the card holds.
                    for p in own.iter().chain(book.take_landed(flight, cover_value).iter().flatten()) {
                        title[*p as usize] = card[*p as usize];
                        book.found(1, *p, flight);
                    }
                    for (p, v) in writes {
                        assert!(title[p as usize] >= v, "D {deadline}: flight {flight}'s write {v} of piece {p} is not in title memory");
                    }
                    landings += 1;
                }
            }
            assert!(landings > 100, "D {deadline}: {landings} landings");
        }
    }
}
