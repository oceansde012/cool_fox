//! Clean-up of a finished module before it is written: the lowering emits
//! one instruction after another as the console code names them, and
//! nothing stands between it and the driver's compiler. Every pass here is
//! exact: the module computes the same values and has the same effects for
//! every input.
//!
//! - [`DEAD`]: instructions without an effect whose result nothing reads
//!   are left out (the lowering computes every dword of every register a
//!   console instruction writes; a guarded write reads the register's old
//!   value whether anything reads the new one or not).
//! - [`FOLD`]: integer arithmetic on constants is done here, and identities
//!   (`x + 0`, `x & -1`, a select between a value and itself, a cast there
//!   and back, a join of one value) give their operand. Float arithmetic is
//!   left alone: `x + 0.0` and `x * 1.0` are not `x` for every `x`.
//! - [`REUSE`]: an instruction without an effect whose operands are those
//!   of an earlier one that every path to it has run gives that one's
//!   result (a descriptor handle made again in each block, an index clamped
//!   again, a register read as a float again).
//! - [`FACTS`]: in the blocks only one side of a branch leads to, the
//!   branch's condition is that side's constant, and so is what the
//!   condition was made of (both terms of an `and` that held, the opposite
//!   comparison). The lowering branches on a lane's condition and then
//!   guards each instruction behind the branch with the same condition.
//! - [`LOADS`]: within a block, a read of the same place as an earlier read
//!   gives that one's result while no store, atomic, barrier or other call
//!   with an effect stands between them (another thread's unsynchronised
//!   store may fall between any two reads or before both: the title's code
//!   cannot tell which).
//! - [`EXT_FOLD`]: opt-in extended algebraic, bitwise, shift, power-of-two
//!   strength reduction, comparison/select folds, and canonical comparison
//!   reuse and branch facts.
//!
//! Which passes run is `emit::shader_opt()` (`VICTORY_SHADER_OPT`), part of
//! the translation and so of the shader cache's key.

use std::collections::HashMap;

use super::{
    Attr, Bin, Cast, Const, Func, MdDef, Module, Op, Term, Ty, TypeDef, V, for_each_operand, pred,
    successors,
};

/// Dead-code elimination.
pub const DEAD: u32 = 1;
/// Constants folded, identities removed.
pub const FOLD: u32 = 2;
/// One instruction for one value.
pub const REUSE: u32 = 4;
/// What a branch taken says about its condition.
pub const FACTS: u32 = 8;
/// One read for one place within a block, while nothing is stored.
pub const LOADS: u32 = 16;
/// Not a pass over the module: the lowering of a first-lane read in a
/// wave64 program run as two host waves skips the exchange that asks which
/// half has a lane where every lane is in EXEC (`lower.rs`,
/// `v_readfirstlane_b32`).
pub const FIRST: u32 = 32;
/// Not a pass over the module either: the lowering reads the dwords of one
/// console load (`s_buffer_load_dwordx2` to `x16` from an SRV, a
/// `global_load_dwordx2` to `x4`) by one call for every four of them, not
/// by a call each (`lower.rs`, `raw_dwords`). The same bytes are read.
/// From fxceless/victory-fork c36971b2. On an RX 7900 XTX it is worth nothing:
/// the yard's frame replayed turn by turn, dispatches and draws 18.47 -> 18.45
/// ms (12 loops) and 18.23 -> 18.16 (5 loops): the driver joins the loads.
pub const WIDE: u32 = 64;
/// Opt-in extended algebraic, bitwise, shift, compare/select folds, and
/// canonical comparison reuse and branch facts.
pub const EXT_FOLD: u32 = 128;
/// The passes over the module.
pub const MODULE: u32 = DEAD | FOLD | REUSE | FACTS | LOADS | EXT_FOLD;
/// Every pass.
pub const ALL: u32 = MODULE | FIRST | WIDE;
/// The passes that run when nothing says which: the original default passes
/// (`63`), keeping default output bit-identical unless opt-in bits are set.
pub const DEFAULT: u32 = DEAD | FOLD | REUSE | FACTS | LOADS | FIRST;

/// What the instructions a module writes use, of the features a shader
/// declares by flag (the validator's "flags must match usage").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Usage {
    /// A wave or quad operation.
    pub wave_ops: bool,
    /// A rasterizer ordered view.
    pub rovs: bool,
    /// A 64-bit integer, as a result or an operand.
    pub int64: bool,
    /// A typed load of several components from a UAV.
    pub typed_uav_loads: bool,
    /// A handle from the view heap.
    pub heap_views: bool,
    /// A handle from the sampler heap.
    pub heap_samplers: bool,
    /// A 64-bit atomic.
    pub atomic64: bool,
}

/// What makes two instructions without an effect the same value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Key {
    Bin(u8, V, V),
    Cast(u8, Ty, V),
    Cmp(u8, V, V),
    Select(V, V, V),
    Extract(V, u32),
    Call(u32, Vec<V>),
    Gep(Ty, V, Vec<V>),
    Load(V),
}

/// The blocks reached from the entry, each after the blocks that dominate
/// it, and each block's children in the dominator tree.
struct Tree {
    order: Vec<usize>,
    children: Vec<Vec<usize>>,
    /// Per block: the reached blocks that branch to it, once per edge.
    preds: Vec<Vec<usize>>,
}

/// The comparison that holds exactly when `p` does not (floats 0-15,
/// integers 32-41).
fn opposite(p: u8) -> u8 {
    match p {
        0..=15 => 15 - p,
        pred::EQ => pred::NE,
        pred::NE => pred::EQ,
        pred::IUGT => pred::IULE,
        pred::IULE => pred::IUGT,
        pred::IUGE => pred::IULT,
        pred::IULT => pred::IUGE,
        pred::SGT => pred::SLE,
        pred::SLE => pred::SGT,
        pred::SGE => pred::SLT,
        _ => pred::SGE,
    }
}

/// Canonical `(pred, a, b)` for comparison equivalence: symmetric predicates
/// order operands `a <= b`, and greater-than predicates flip to less-than by
/// swapping `(a, b)`. Preserves IEEE-754 NaN semantics for every float
/// predicate (ordered stays ordered, unordered stays unordered).
fn canonical_cmp(p: u8, a: V, b: V) -> (u8, V, V) {
    match p {
        pred::EQ
        | pred::NE
        | pred::OEQ
        | pred::UEQ
        | pred::ONE
        | pred::UNE
        | pred::ORD
        | pred::UNO => {
            if b < a {
                (p, b, a)
            } else {
                (p, a, b)
            }
        }
        pred::IUGT => (pred::IULT, b, a),
        pred::IUGE => (pred::IULE, b, a),
        pred::SGT => (pred::SLT, b, a),
        pred::SGE => (pred::SLE, b, a),
        pred::OGT => (pred::OLT, b, a),
        pred::OGE => (pred::OLE, b, a),
        pred::UGT => (pred::ULT, b, a),
        pred::UGE => (pred::ULE, b, a),
        _ => (p, a, b),
    }
}

/// What holds in the block being walked: values that are a known bit there,
/// and comparisons (by their operands) with a known outcome.
#[derive(Default)]
struct Facts {
    bits: HashMap<V, bool>,
    compares: HashMap<(u8, V, V), bool>,
}

/// The facts a block added, to forget when its part of the tree is done.
#[derive(Default)]
struct Learned {
    bits: Vec<V>,
    compares: Vec<(u8, V, V)>,
}

fn mask(bits: u32) -> u64 {
    if bits >= 64 { !0 } else { (1 << bits) - 1 }
}

/// `x` of `bits` bits as a signed number.
fn signed(bits: u32, x: u64) -> i64 {
    if bits < 64 && x >> (bits - 1) & 1 != 0 {
        (x | !mask(bits)) as i64
    } else {
        x as i64
    }
}

/// What `v` stands for after the replacements so far.
fn resolve(repl: &[Option<V>], mut v: V) -> V {
    while let V::Inst(i) = v {
        match repl[i as usize] {
            Some(r) => v = r,
            None => break,
        }
    }
    v
}

/// Replaces every value `op` reads by `f` of it.
fn map_operands(op: &mut Op, mut f: impl FnMut(V) -> V) {
    match op {
        Op::Bin(_, a, b) | Op::Cmp(_, a, b) | Op::Store(a, b) | Op::AtomicRmw(_, a, b) => {
            *a = f(*a);
            *b = f(*b);
        }
        Op::Cast(_, v) | Op::Extract(v, _) | Op::Load(v) => *v = f(*v),
        Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => {
            *a = f(*a);
            *b = f(*b);
            *c = f(*c);
        }
        Op::Call(_, args) => args.iter_mut().for_each(|a| *a = f(*a)),
        Op::Phi(inc) => inc.iter_mut().for_each(|x| x.0 = f(x.0)),
        Op::Gep(_, p, idx) => {
            *p = f(*p);
            idx.iter_mut().for_each(|i| *i = f(*i));
        }
    }
}

/// What a call does besides returning its result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Effect {
    /// Nothing, and the result depends on the arguments alone for the whole
    /// run of the shader (arithmetic, handles, inputs, constant buffers,
    /// resource sizes).
    Pure,
    /// Nothing, but the result depends on memory a store may change
    /// (buffer and texture reads).
    Reads,
    /// Nothing, but the result depends on which lanes run the instruction
    /// (wave and quad operations).
    Lanes,
    /// Stores, atomics, barriers, outputs, discards: always kept.
    Writes,
}

pub(super) fn effect(f: &Func) -> Effect {
    let name = f.name.strip_prefix("dx.op.").unwrap_or(&f.name);
    match f.attr {
        Some(Attr::ReadNone) => Effect::Pure,
        // A shader cannot write a constant buffer or resize a resource.
        Some(Attr::ReadOnly)
            if name.starts_with("cbufferLoadLegacy") || name.starts_with("getDimensions") =>
        {
            Effect::Pure
        }
        Some(Attr::ReadOnly) if name.starts_with("wave") => Effect::Lanes,
        Some(Attr::ReadOnly) => Effect::Reads,
        // Declared with the attribute of calls that have effects, and have
        // none: they read the other lanes' values.
        _ if name.starts_with("wave") || name.starts_with("quad") => Effect::Lanes,
        _ => Effect::Writes,
    }
}

impl Module {
    /// Runs the clean-up passes named in `passes` ([`ALL`]: every one).
    pub(crate) fn optimise(&mut self, passes: u32) {
        if passes & (FOLD | REUSE | FACTS | LOADS | EXT_FOLD) != 0 {
            // Values merged in one round can make two joins the same in
            // the next (a loop's registers that only go round), and a
            // branch that goes one way leaves blocks with one way in.
            for _ in 0..4 {
                let mut changed = self.merge(passes);
                if passes & (FOLD | EXT_FOLD) != 0 {
                    changed |= self.prune();
                }
                if !changed {
                    break;
                }
            }
        }
        if passes & DEAD != 0 {
            self.drop_dead();
        }
    }

    /// Per function: what a call of it does.
    pub(super) fn effects(&self) -> Vec<Effect> {
        self.funcs.iter().map(effect).collect()
    }

    /// The blocks reached from the entry.
    pub(super) fn reached(&self) -> Vec<bool> {
        let mut seen = vec![false; self.blocks.len()];
        if self.blocks.is_empty() {
            return seen;
        }
        let mut stack = vec![0usize];
        seen[0] = true;
        while let Some(b) = stack.pop() {
            for &s in successors(self.blocks[b].term).iter() {
                if !seen[s] {
                    seen[s] = true;
                    stack.push(s);
                }
            }
        }
        seen
    }

    fn dominator_tree(&self) -> Tree {
        let n = self.blocks.len();
        let succs = |b: usize| successors(self.blocks[b].term);
        // Reverse postorder, as `write` lays the blocks out.
        let mut seen = vec![false; n];
        let mut post = Vec::new();
        let mut stack = vec![(0usize, 0usize)];
        seen[0] = true;
        while let Some(top) = stack.last_mut() {
            let (b, k) = *top;
            top.1 += 1;
            match succs(b).get(k) {
                Some(&s) if !seen[s] => {
                    seen[s] = true;
                    stack.push((s, 0));
                }
                Some(_) => {}
                None => {
                    post.push(b);
                    stack.pop();
                }
            }
        }
        post.reverse();
        let order = post;
        let mut number = vec![usize::MAX; n];
        for (k, &b) in order.iter().enumerate() {
            number[b] = k;
        }
        let mut preds = vec![Vec::new(); n];
        for &b in &order {
            for &s in succs(b).iter() {
                preds[s].push(b);
            }
        }
        // Immediate dominators (Cooper, Harvey and Kennedy's iteration).
        let mut idom = vec![usize::MAX; n];
        idom[0] = 0;
        let mut changed = true;
        while changed {
            changed = false;
            for &b in &order[1..] {
                let mut new = usize::MAX;
                for &p in &preds[b] {
                    if idom[p] == usize::MAX {
                        continue;
                    }
                    if new == usize::MAX {
                        new = p;
                        continue;
                    }
                    let (mut x, mut y) = (p, new);
                    while x != y {
                        while number[x] > number[y] {
                            x = idom[x];
                        }
                        while number[y] > number[x] {
                            y = idom[y];
                        }
                    }
                    new = x;
                }
                if idom[b] != new {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        let mut children = vec![Vec::new(); n];
        for &b in &order[1..] {
            children[idom[b]].push(b);
        }
        Tree {
            order,
            children,
            preds,
        }
    }

    /// Notes that `v` (an i1) is `holds` in the block being walked, and what
    /// follows from it for the values `v` was made of.
    #[allow(clippy::too_many_arguments)]
    fn learn(
        &self,
        repl: &[Option<V>],
        v: V,
        holds: bool,
        facts: &mut Facts,
        learned: &mut Learned,
        depth: u32,
        ext_fold: bool,
    ) {
        let v = if ext_fold { resolve(repl, v) } else { v };
        let V::Inst(i) = v else { return };
        if facts.bits.contains_key(&v) {
            return;
        }
        facts.bits.insert(v, holds);
        learned.bits.push(v);
        if depth == 0 {
            return;
        }
        match self.insts[i as usize].op {
            // Both terms of an `and` that holds hold; neither of an `or`
            // that does not.
            Op::Bin(Bin::And, a, b) if holds => {
                self.learn(repl, a, true, facts, learned, depth - 1, ext_fold);
                self.learn(repl, b, true, facts, learned, depth - 1, ext_fold);
            }
            Op::Bin(Bin::Or, a, b) if !holds => {
                self.learn(repl, a, false, facts, learned, depth - 1, ext_fold);
                self.learn(repl, b, false, facts, learned, depth - 1, ext_fold);
            }
            Op::Bin(Bin::Xor, a, b) => {
                let (a, b) = if ext_fold {
                    (resolve(repl, a), resolve(repl, b))
                } else {
                    (a, b)
                };
                let (x, k) = match (self.int_k(a), self.int_k(b)) {
                    (None, Some((1, k))) => (a, k),
                    (Some((1, k)), None) => (b, k),
                    _ => return,
                };
                self.learn(repl, x, holds != (k != 0), facts, learned, depth - 1, ext_fold);
            }
            // The opposite comparison of the same operands (and, under
            // `EXT_FOLD`, canonicalised so flipped operands hit the same entry).
            Op::Cmp(p, a, b) => {
                if ext_fold {
                    let (a, b) = (resolve(repl, a), resolve(repl, b));
                    for (key, val) in [
                        (canonical_cmp(p, a, b), holds),
                        (canonical_cmp(opposite(p), a, b), !holds),
                    ] {
                        if let std::collections::hash_map::Entry::Vacant(e) =
                            facts.compares.entry(key)
                        {
                            e.insert(val);
                            learned.compares.push(key);
                        }
                    }
                } else {
                    let key = (opposite(p), a, b);
                    if let std::collections::hash_map::Entry::Vacant(e) =
                        facts.compares.entry(key)
                    {
                        e.insert(!holds);
                        learned.compares.push(key);
                    }
                }
            }
            _ => {}
        }
    }

    fn int_bits(&self, ty: Ty) -> Option<u32> {
        match self.types[ty.0 as usize] {
            TypeDef::Int(bits) => Some(bits),
            _ => None,
        }
    }

    /// An integer constant's width and value (the bits of that width).
    pub(crate) fn int_k(&self, v: V) -> Option<(u32, u64)> {
        let V::Const(i) = v else { return None };
        let (ty, Const::Int(x)) = &self.consts[i as usize] else {
            return None;
        };
        let bits = self.int_bits(*ty)?;
        Some((bits, *x as u64 & mask(bits)))
    }

    /// The bits of a 32-bit float that is a constant, given as one or as a
    /// cast of an integer constant.
    pub(crate) fn float_k(&self, v: V) -> Option<f32> {
        let (v, cast) = match v {
            V::Inst(i) => match self.insts[i as usize].op {
                Op::Cast(Cast::Bitcast, x) => (x, true),
                _ => return None,
            },
            _ => (v, false),
        };
        let V::Const(i) = v else { return None };
        match &self.consts[i as usize].1 {
            Const::Float(bits) if !cast => Some(f32::from_bits(*bits)),
            Const::Int(_) if cast => {
                let (bits, x) = self.int_k(v)?;
                (bits == 32).then_some(f32::from_bits(x as u32))
            }
            _ => None,
        }
    }

    /// For a float that is a number and not an infinity whatever the program
    /// reads: a power of two its magnitude does not pass.
    pub(crate) fn bounded(&self, v: V, depth: u32) -> Option<f32> {
        if let Some(k) = self.float_k(v) {
            return k.is_finite().then(|| k.abs().max(1e-30).log2().ceil());
        }
        let V::Inst(i) = v else { return None };
        let depth = depth.checked_sub(1)?;
        let of = |v: V| self.bounded(v, depth);
        // (A power that overflows, or might with rounding, is no bound.)
        let held = |b: f32| (b < 126.0).then_some(b);
        match &self.insts[i as usize].op {
            Op::Cast(Cast::UiToFp | Cast::SiToFp, _) => Some(32.0),
            Op::Cast(Cast::Bitcast, x) => match *x {
                V::Inst(j) => match self.insts[j as usize].op {
                    Op::Cast(Cast::Bitcast, y) if self.type_of(y) == self.type_of(v) => of(y),
                    _ => None,
                },
                _ => None,
            },
            Op::Bin(Bin::Mul, a, b) => held(of(*a)? + of(*b)?),
            Op::Bin(Bin::Add | Bin::Sub, a, b) => held(of(*a)?.max(of(*b)?) + 1.0),
            Op::Select(_, a, b) => Some(of(*a)?.max(of(*b)?)),
            Op::Call(_, args) => {
                let (_, code) = self.int_k(*args.first()?)?;
                match (code, &args[1..]) {
                    // Saturated, a fraction, a cosine or sine (of a number).
                    (7, [_]) => Some(0.0),
                    (6, [x]) => of(*x),
                    (22 | 12 | 13, [x]) => of(*x).map(|_| 0.0),
                    (26..=29, [x]) => held(of(*x)? + 1.0),
                    // Of two numbers; or one of them the other way round: the
                    // minimum (36) of a constant and a maximum of another.
                    (35 | 36, [a, b]) => match (of(*a), of(*b)) {
                        (Some(x), Some(y)) => Some(x.max(y)),
                        _ => {
                            let other = if code == 35 { 36 } else { 35 };
                            let (inner, k) = if self.float_k(*b).is_some() { (*a, *b) } else { (*b, *a) };
                            let k = self.float_k(k).and_then(|_| of(k))?;
                            let V::Inst(j) = inner else { return None };
                            let Op::Call(_, inner) = &self.insts[j as usize].op else { return None };
                            let (_, code) = self.int_k(*inner.first()?)?;
                            let low = inner[1..].iter().find(|x| self.float_k(**x).is_some())?;
                            (code == other).then_some(())?;
                            Some(k.max(of(*low)?))
                        }
                    },
                    (46, [a, b, c]) => held((of(*a)? + of(*b)?).max(of(*c)?) + 1.0),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// The constant `x` of the integer type `ty`, kept sign-extended as
    /// [`Module::i32`] and [`Module::bool`] keep theirs.
    fn k(&mut self, ty: Ty, bits: u32, x: u64) -> V {
        self.int_const(ty, signed(bits, x & mask(bits)))
    }

    /// If `v` is `xor(x, true)` or `xor(true, x)` on `i1`, returns `Some(x)`.
    fn is_bool_not(&self, v: V) -> Option<V> {
        let V::Inst(j) = v else { return None };
        let inst = &self.insts[j as usize];
        if self.int_bits(inst.ty) != Some(1) {
            return None;
        }
        let Op::Bin(Bin::Xor, a, b) = inst.op else {
            return None;
        };
        match (self.int_k(a), self.int_k(b)) {
            (None, Some((1, 1))) => Some(a),
            (Some((1, 1)), None) => Some(b),
            _ => None,
        }
    }

    /// Simplifies `xor(x, true)` for an `i1` instruction `i`: returns a direct
    /// replacement value when possible, or rewrites `self.insts[i].op` in place
    /// to an inverted comparison or `xor(x, true)`.
    fn negate_bool_in_place(&mut self, i: u32, x: V, changed: &mut bool) -> Option<V> {
        if let Some((1, k)) = self.int_k(x) {
            return Some(self.bool(k == 0));
        }
        if let Some(orig) = self.is_bool_not(x) {
            return Some(orig);
        }
        if let V::Inst(j) = x
            && let Op::Cmp(p, a, b) = self.insts[j as usize].op
            && (p <= pred::TRUE || (pred::EQ..=pred::SLE).contains(&p))
        {
            let opp = opposite(p);
            if opp == pred::FALSE {
                return Some(self.bool(false));
            }
            if opp == pred::TRUE {
                return Some(self.bool(true));
            }
            // Never emit `fcmp one` (`pred::ONE`): the AMD D3D12 driver
            // treats it as `une`.
            if opp != pred::ONE {
                self.insts[i as usize].op = Op::Cmp(opp, a, b);
                *changed = true;
                return None;
            }
        }
        let yes = self.bool(true);
        let already = matches!(
            self.insts[i as usize].op,
            Op::Bin(Bin::Xor, a, b) if (a == x && b == yes) || (a == yes && b == x)
        );
        if !already {
            self.insts[i as usize].op = Op::Bin(Bin::Xor, x, yes);
            *changed = true;
        }
        None
    }

    /// If `v` is an integer instruction of form `(x op c)` (with either operand
    /// order for commutative `op`), returns `Some((x, c))`.
    fn bin_with_const(&self, v: V, want: Bin) -> Option<(V, u64)> {
        let V::Inst(j) = v else { return None };
        let Op::Bin(op, a, b) = self.insts[j as usize].op else {
            return None;
        };
        if op != want {
            return None;
        }
        match (self.int_k(a), self.int_k(b)) {
            (None, Some((_, k))) => Some((a, k)),
            (Some((_, k)), None)
                if matches!(op, Bin::Add | Bin::Mul | Bin::And | Bin::Or | Bin::Xor) =>
            {
                Some((b, k))
            }
            _ => None,
        }
    }

    /// If `v` is a two-valued integer driven by an `i1` condition `c`, returns
    /// `(c, value_when_true, value_when_false)` masked to `bits`.
    fn two_valued_int(&self, v: V, bits: u32) -> Option<(V, u64, u64)> {
        if bits == 1 && self.int_k(v).is_none() {
            return Some((v, 1, 0));
        }
        let V::Inst(j) = v else { return None };
        match self.insts[j as usize].op {
            Op::Select(c, t, f) => {
                let (_, k1) = self.int_k(t)?;
                let (_, k2) = self.int_k(f)?;
                Some((c, k1, k2))
            }
            Op::Cast(Cast::ZExt, b) if self.int_bits(self.type_of(b)) == Some(1) => {
                Some((b, 1, 0))
            }
            Op::Cast(Cast::SExt, b) if self.int_bits(self.type_of(b)) == Some(1) => {
                Some((b, mask(bits), 0))
            }
            _ => None,
        }
    }

    /// The value instruction `i` is known to have without it, if any: a
    /// constant or an earlier value. May instead give `i` simpler operands
    /// (`changed` says so). When `ext_fold` is false, runs only the base
    /// [`FOLD`] rules so [`DEFAULT`] stays bit-identical.
    fn simplify(&mut self, i: u32, ext_fold: bool, changed: &mut bool) -> Option<V> {
        let ty = self.insts[i as usize].ty;
        match self.insts[i as usize].op {
            Op::Bin(op, a, b) => {
                // Floats: `x + 0.0`, `x * 1.0` and the like are not `x` for
                // every `x` (a negative zero, a signalling NaN).
                let bits = self.int_bits(ty)?;
                let (ka, kb) = (self.int_k(a).map(|k| k.1), self.int_k(b).map(|k| k.1));
                let all = mask(bits);
                if let (Some(x), Some(y)) = (ka, kb) {
                    // A shift by the width or more and a division by zero
                    // (or signed MIN / -1) have no defined result: left as they are.
                    let amount = u32::try_from(y).ok().filter(|&y| y < bits);
                    let r = match op {
                        Bin::Add => x.wrapping_add(y),
                        Bin::Sub => x.wrapping_sub(y),
                        Bin::Mul => x.wrapping_mul(y),
                        Bin::And => x & y,
                        Bin::Or => x | y,
                        Bin::Xor => x ^ y,
                        Bin::Shl => x << amount?,
                        Bin::LShr => x >> amount?,
                        Bin::AShr => (signed(bits, x) >> amount?) as u64,
                        Bin::UDiv if y != 0 => x / y,
                        Bin::SDiv
                            if ext_fold
                                && y != 0
                                && !(signed(bits, y) == -1 && x == (1u64 << (bits - 1))) =>
                        {
                            (signed(bits, x) / signed(bits, y)) as u64
                        }
                        Bin::URem if y != 0 => x % y,
                        _ => return None,
                    };
                    return Some(self.k(ty, bits, r));
                }
                match (op, ka, kb) {
                    (
                        Bin::Add | Bin::Sub | Bin::Or | Bin::Xor | Bin::Shl | Bin::LShr | Bin::AShr,
                        _,
                        Some(0),
                    ) => return Some(a),
                    (Bin::Add | Bin::Or | Bin::Xor, Some(0), _) => return Some(b),
                    (Bin::Mul | Bin::UDiv, _, Some(1)) => return Some(a),
                    (Bin::SDiv, _, Some(1)) if ext_fold && bits > 1 => return Some(a),
                    (Bin::Mul, Some(1), _) => return Some(b),
                    (Bin::Mul | Bin::And, _, Some(0)) => return Some(b),
                    (Bin::Mul | Bin::And, Some(0), _) => return Some(a),
                    (Bin::URem, _, Some(1)) if ext_fold => return Some(self.k(ty, bits, 0)),
                    (Bin::And, _, Some(m)) if m == all => return Some(a),
                    (Bin::And, Some(m), _) if m == all => return Some(b),
                    (Bin::Or, _, Some(m)) if m == all => return Some(b),
                    (Bin::Or, Some(m), _) if m == all => return Some(a),
                    (Bin::And | Bin::Or, _, _) if a == b => return Some(a),
                    (Bin::Sub | Bin::Xor, _, _) if ext_fold && a == b => {
                        return Some(self.k(ty, bits, 0));
                    }
                    _ => {}
                }
                if !ext_fold {
                    return None;
                }
                // Extended algebraic, bitwise, shift-mask, and power-of-two strength reduction.
                match op {
                    Bin::Xor => {
                        if bits == 1 {
                            match (ka, kb) {
                                (None, Some(1)) => return self.negate_bool_in_place(i, a, changed),
                                (Some(1), None) => return self.negate_bool_in_place(i, b, changed),
                                _ => {}
                            }
                        }
                        let const_pair = match (ka, kb) {
                            (None, Some(c2)) => Some((a, c2)),
                            (Some(c2), None) => Some((b, c2)),
                            _ => None,
                        };
                        if let Some((other, c2)) = const_pair
                            && let Some((x, c1)) = self.bin_with_const(other, Bin::Xor)
                        {
                            let c = (c1 ^ c2) & all;
                            if c == 0 {
                                return Some(x);
                            }
                            let ck = self.k(ty, bits, c);
                            self.insts[i as usize].op = Op::Bin(Bin::Xor, x, ck);
                            *changed = true;
                            return None;
                        }
                        // `(x ^ y) ^ y => x` and symmetric forms.
                        if let V::Inst(j) = a
                            && let Op::Bin(Bin::Xor, x, y) = self.insts[j as usize].op
                        {
                            if y == b {
                                return Some(x);
                            }
                            if x == b {
                                return Some(y);
                            }
                        }
                        if let V::Inst(j) = b
                            && let Op::Bin(Bin::Xor, x, y) = self.insts[j as usize].op
                        {
                            if y == a {
                                return Some(x);
                            }
                            if x == a {
                                return Some(y);
                            }
                        }
                        None
                    }
                    Bin::Add => {
                        let const_pair = match (ka, kb) {
                            (None, Some(c2)) => Some((a, c2)),
                            (Some(c2), None) => Some((b, c2)),
                            _ => None,
                        };
                        if let Some((other, c2)) = const_pair {
                            if let Some((x, c1)) = self.bin_with_const(other, Bin::Add) {
                                let c = c1.wrapping_add(c2) & all;
                                if c == 0 {
                                    return Some(x);
                                }
                                let ck = self.k(ty, bits, c);
                                self.insts[i as usize].op = Op::Bin(Bin::Add, x, ck);
                                *changed = true;
                                return None;
                            }
                            if let Some((x, c1)) = self.bin_with_const(other, Bin::Sub) {
                                let d = c2.wrapping_sub(c1) & all;
                                if d == 0 {
                                    return Some(x);
                                }
                                let dk = self.k(ty, bits, d);
                                self.insts[i as usize].op = Op::Bin(Bin::Add, x, dk);
                                *changed = true;
                                return None;
                            }
                        }
                        // `(x - y) + y => x` and `y + (x - y) => x`.
                        if let V::Inst(j) = a
                            && let Op::Bin(Bin::Sub, x, y) = self.insts[j as usize].op
                            && y == b
                        {
                            return Some(x);
                        }
                        if let V::Inst(j) = b
                            && let Op::Bin(Bin::Sub, x, y) = self.insts[j as usize].op
                            && y == a
                        {
                            return Some(x);
                        }
                        None
                    }
                    Bin::Sub => {
                        if let Some(c2) = kb {
                            if let Some((x, c1)) = self.bin_with_const(a, Bin::Sub) {
                                let c = c1.wrapping_add(c2) & all;
                                if c == 0 {
                                    return Some(x);
                                }
                                let ck = self.k(ty, bits, c);
                                self.insts[i as usize].op = Op::Bin(Bin::Sub, x, ck);
                                *changed = true;
                                return None;
                            }
                            if let Some((x, c1)) = self.bin_with_const(a, Bin::Add) {
                                let d = c1.wrapping_sub(c2) & all;
                                if d == 0 {
                                    return Some(x);
                                }
                                let dk = self.k(ty, bits, d);
                                self.insts[i as usize].op = Op::Bin(Bin::Add, x, dk);
                                *changed = true;
                                return None;
                            }
                        }
                        // `(x + y) - y => x` and `(y + x) - y => x`.
                        if let V::Inst(j) = a
                            && let Op::Bin(Bin::Add, x, y) = self.insts[j as usize].op
                        {
                            if y == b {
                                return Some(x);
                            }
                            if x == b {
                                return Some(y);
                            }
                        }
                        None
                    }
                    Bin::And => {
                        let const_pair = match (ka, kb) {
                            (None, Some(c2)) => Some((a, c2)),
                            (Some(c2), None) => Some((b, c2)),
                            _ => None,
                        };
                        if let Some((other, c2)) = const_pair {
                            if let Some((x, c1)) = self.bin_with_const(other, Bin::And) {
                                let c = c1 & c2;
                                if c == 0 {
                                    return Some(self.k(ty, bits, 0));
                                }
                                if c == c1 {
                                    return Some(other);
                                }
                                let ck = self.k(ty, bits, c);
                                self.insts[i as usize].op = Op::Bin(Bin::And, x, ck);
                                *changed = true;
                                return None;
                            }
                            // `and(zext i1 b, c)` is `zext i1 b` when bit 0 is set, else 0.
                            if let V::Inst(z) = other
                                && let Op::Cast(Cast::ZExt, src) = self.insts[z as usize].op
                                && self.int_bits(self.type_of(src)) == Some(1)
                            {
                                return Some(if c2 & 1 != 0 {
                                    other
                                } else {
                                    self.k(ty, bits, 0)
                                });
                            }
                        }
                        None
                    }
                    Bin::Or => {
                        let const_pair = match (ka, kb) {
                            (None, Some(c2)) => Some((a, c2)),
                            (Some(c2), None) => Some((b, c2)),
                            _ => None,
                        };
                        if let Some((other, c2)) = const_pair
                            && let Some((x, c1)) = self.bin_with_const(other, Bin::Or)
                        {
                            let c = (c1 | c2) & all;
                            if c == all {
                                return Some(self.k(ty, bits, all));
                            }
                            if c == c1 {
                                return Some(other);
                            }
                            let ck = self.k(ty, bits, c);
                            self.insts[i as usize].op = Op::Bin(Bin::Or, x, ck);
                            *changed = true;
                            return None;
                        }
                        None
                    }
                    Bin::Shl | Bin::LShr | Bin::AShr => {
                        if let Some(c2) = kb
                            && c2 < u64::from(bits)
                            && let Some((x, c1)) = self.bin_with_const(a, op)
                            && c1 < u64::from(bits)
                            && let Some(sum) = c1.checked_add(c2)
                        {
                            if sum < u64::from(bits) {
                                let sk = self.k(ty, bits, sum);
                                self.insts[i as usize].op = Op::Bin(op, x, sk);
                                *changed = true;
                                return None;
                            }
                            if matches!(op, Bin::Shl | Bin::LShr) {
                                return Some(self.k(ty, bits, 0));
                            }
                            let max_s = u64::from(bits - 1);
                            if c1 == max_s {
                                return Some(a);
                            }
                            let sk = self.k(ty, bits, max_s);
                            self.insts[i as usize].op = Op::Bin(Bin::AShr, x, sk);
                            *changed = true;
                            return None;
                        }
                        None
                    }
                    Bin::Mul => {
                        let pow2 = match (ka, kb) {
                            (None, Some(c)) if c > 1 && c.is_power_of_two() => Some((a, c)),
                            (Some(c), None) if c > 1 && c.is_power_of_two() => Some((b, c)),
                            _ => None,
                        };
                        if let Some((x, c)) = pow2 {
                            let sh = self.k(ty, bits, u64::from(c.trailing_zeros()));
                            self.insts[i as usize].op = Op::Bin(Bin::Shl, x, sh);
                            *changed = true;
                        }
                        None
                    }
                    Bin::UDiv => {
                        if let Some(c) = kb
                            && c > 1
                            && c.is_power_of_two()
                        {
                            let sh = self.k(ty, bits, u64::from(c.trailing_zeros()));
                            self.insts[i as usize].op = Op::Bin(Bin::LShr, a, sh);
                            *changed = true;
                        }
                        None
                    }
                    Bin::URem => {
                        if let Some(c) = kb
                            && c > 1
                            && c.is_power_of_two()
                        {
                            let m = self.k(ty, bits, c - 1);
                            self.insts[i as usize].op = Op::Bin(Bin::And, a, m);
                            *changed = true;
                        }
                        None
                    }
                    _ => None,
                }
            }
            Op::Cmp(p, a, b) if p <= pred::TRUE => {
                if ext_fold {
                    if p == pred::FALSE {
                        return Some(self.bool(false));
                    }
                    if p == pred::TRUE {
                        return Some(self.bool(true));
                    }
                }
                // Two float constants compare one way (`v_mul_legacy` asks
                // whether its constant factor is zero).
                let (V::Const(x), V::Const(y)) = (a, b) else {
                    return None;
                };
                let (Const::Float(x), Const::Float(y)) =
                    (&self.consts[x as usize].1, &self.consts[y as usize].1)
                else {
                    return None;
                };
                let (x, y) = (f32::from_bits(*x), f32::from_bits(*y));
                let unordered = x.is_nan() || y.is_nan();
                // Bits 0-2 of a predicate: equal, greater, less; bit 3:
                // or unordered.
                let holds = if unordered {
                    p & 8 != 0
                } else {
                    (p & 1 != 0 && x == y) || (p & 2 != 0 && x > y) || (p & 4 != 0 && x < y)
                };
                Some(self.bool(holds))
            }
            Op::Cmp(p, a, b) => {
                if !(pred::EQ..=pred::SLE).contains(&p) {
                    return None;
                }
                let holds = |x: u64, y: u64, bits: u32| {
                    let (sx, sy) = (signed(bits, x), signed(bits, y));
                    match p {
                        pred::EQ => x == y,
                        pred::NE => x != y,
                        pred::IUGT => x > y,
                        pred::IUGE => x >= y,
                        pred::IULT => x < y,
                        pred::IULE => x <= y,
                        pred::SGT => sx > sy,
                        pred::SGE => sx >= sy,
                        pred::SLT => sx < sy,
                        _ => sx <= sy,
                    }
                };
                if let (Some((bits, x)), Some((_, y))) = (self.int_k(a), self.int_k(b)) {
                    return Some(self.bool(holds(x, y, bits)));
                }
                if a == b {
                    return Some(self.bool(holds(0, 0, 32)));
                }
                if !ext_fold {
                    // A bit widened and tested again is the bit.
                    if p == pred::NE
                        && self.int_k(b).is_some_and(|k| k.1 == 0)
                        && let V::Inst(z) = a
                        && let Op::Cast(Cast::ZExt, x) = self.insts[z as usize].op
                        && self.type_of(x) == ty
                    {
                        return Some(x);
                    }
                    return None;
                }
                // Comparing `select(c, k1, k2)`, `zext i1 c`, `sext i1 c`, or
                // `i1` against an integer constant reduces directly to a
                // constant, `c`, or `!c`.
                let reduced = if let Some((bits, k)) = self.int_k(b)
                    && let Some((c, k1, k2)) = self.two_valued_int(a, bits)
                {
                    Some((c, holds(k1, k, bits), holds(k2, k, bits)))
                } else if let Some((bits, k)) = self.int_k(a)
                    && let Some((c, k1, k2)) = self.two_valued_int(b, bits)
                {
                    Some((c, holds(k, k1, bits), holds(k, k2, bits)))
                } else {
                    None
                };
                if let Some((c, wt, wf)) = reduced {
                    return match (wt, wf) {
                        (true, true) => Some(self.bool(true)),
                        (false, false) => Some(self.bool(false)),
                        (true, false) => Some(c),
                        (false, true) => self.negate_bool_in_place(i, c, changed),
                    };
                }
                None
            }
            Op::Select(mut c, mut t, mut f) => {
                let mut inner = false;
                if ext_fold {
                    while let Some(not_c) = self.is_bool_not(c) {
                        c = not_c;
                        std::mem::swap(&mut t, &mut f);
                        inner = true;
                    }
                }
                if let Some((_, c)) = self.int_k(c) {
                    return Some(if c != 0 { t } else { f });
                }
                // Under the same condition an inner select is one of its
                // operands (two guarded writes of one register in a row).
                while let V::Inst(x) = t
                    && let Op::Select(c2, t2, _) = self.insts[x as usize].op
                    && c2 == c
                {
                    (t, inner) = (t2, true);
                }
                while let V::Inst(x) = f
                    && let Op::Select(c2, _, f2) = self.insts[x as usize].op
                    && c2 == c
                {
                    (f, inner) = (f2, true);
                }
                if t == f {
                    return Some(t);
                }
                if self.int_bits(ty) == Some(1) {
                    if !ext_fold {
                        if self.int_k(t).is_some_and(|k| k.1 == 1)
                            && self.int_k(f).is_some_and(|k| k.1 == 0)
                        {
                            return Some(c);
                        }
                    } else {
                        let (kt, kf) = (self.int_k(t).map(|k| k.1), self.int_k(f).map(|k| k.1));
                        match (kt, kf) {
                            (Some(1), Some(0)) => return Some(c),
                            (Some(0), Some(1)) => return self.negate_bool_in_place(i, c, changed),
                            (Some(1), _) => {
                                if f == c {
                                    return Some(c);
                                }
                                self.insts[i as usize].op = Op::Bin(Bin::Or, c, f);
                                *changed = true;
                                return None;
                            }
                            (_, Some(0)) => {
                                if t == c {
                                    return Some(c);
                                }
                                self.insts[i as usize].op = Op::Bin(Bin::And, c, t);
                                *changed = true;
                                return None;
                            }
                            _ => {}
                        }
                        if t == c {
                            self.insts[i as usize].op = Op::Bin(Bin::Or, c, f);
                            *changed = true;
                            return None;
                        }
                        if f == c {
                            self.insts[i as usize].op = Op::Bin(Bin::And, c, t);
                            *changed = true;
                            return None;
                        }
                    }
                }
                if inner {
                    self.insts[i as usize].op = Op::Select(c, t, f);
                    *changed = true;
                }
                None
            }
            Op::Cast(op, x) => {
                let from = self.type_of(x);
                let through = match x {
                    V::Inst(y) => match self.insts[y as usize].op {
                        Op::Cast(op2, z) => Some((op2, z)),
                        _ => None,
                    },
                    _ => None,
                };
                match op {
                    Cast::Bitcast => {
                        if from == ty {
                            return Some(x);
                        }
                        // There and back: the bits are the same.
                        if let Some((Cast::Bitcast, z)) = through
                            && self.type_of(z) == ty
                        {
                            return Some(z);
                        }
                        let V::Const(c) = x else { return None };
                        let to_float = matches!(self.types[ty.0 as usize], TypeDef::Float);
                        match self.consts[c as usize].1 {
                            Const::Int(bits) if to_float && self.int_bits(from) == Some(32) => {
                                Some(self.f32(bits as u32))
                            }
                            Const::Float(bits) if self.int_bits(ty) == Some(32) => {
                                Some(self.int_const(ty, i64::from(bits as i32)))
                            }
                            _ => None,
                        }
                    }
                    Cast::ZExt | Cast::SExt | Cast::Trunc => {
                        let to = self.int_bits(ty)?;
                        if let Some((bits, k)) = self.int_k(x) {
                            let k = if op == Cast::SExt {
                                signed(bits, k) as u64
                            } else {
                                k
                            };
                            return Some(self.k(ty, to, k));
                        }
                        // Widened and narrowed again.
                        if op == Cast::Trunc
                            && let Some((Cast::ZExt | Cast::SExt, z)) = through
                            && self.type_of(z) == ty
                        {
                            return Some(z);
                        }
                        None
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// What kind of instruction `i` is for merging.
    fn class(&self, effects: &[Effect], i: u32) -> Effect {
        match &self.insts[i as usize].op {
            Op::Bin(..)
            | Op::Cast(..)
            | Op::Cmp(..)
            | Op::Select(..)
            | Op::Extract(..)
            | Op::Gep(..) => Effect::Pure,
            Op::Load(p) => {
                // A constant table reads the same whatever is stored.
                let base = match *p {
                    V::Inst(g) => match &self.insts[g as usize].op {
                        Op::Gep(_, base, _) => *base,
                        _ => *p,
                    },
                    _ => *p,
                };
                match base {
                    V::Global(g) if self.globals[g as usize].init.is_some() => Effect::Pure,
                    _ => Effect::Reads,
                }
            }
            Op::Store(..) | Op::AtomicRmw(..) | Op::CmpXchg(..) => Effect::Writes,
            Op::Phi(_) => Effect::Lanes,
            Op::Call(f, _) => effects[*f as usize],
        }
    }

    fn key(&self, i: u32, ext_fold: bool) -> Option<Key> {
        let inst = &self.insts[i as usize];
        Some(match &inst.op {
            Op::Bin(op, a, b) => {
                // Integer sums, products and bit operations whichever way
                // round; float ones as written (which NaN an operation of
                // two NaNs gives is the hardware's choice by operand).
                let commutes = matches!(op, Bin::Add | Bin::Mul | Bin::And | Bin::Or | Bin::Xor)
                    && self.int_bits(inst.ty).is_some();
                let (a, b) = if commutes && b < a {
                    (*b, *a)
                } else {
                    (*a, *b)
                };
                Key::Bin(*op as u8, a, b)
            }
            Op::Cast(op, v) => Key::Cast(*op as u8, inst.ty, *v),
            Op::Cmp(p, a, b) => {
                if ext_fold {
                    let (cp, ca, cb) = canonical_cmp(*p, *a, *b);
                    Key::Cmp(cp, ca, cb)
                } else {
                    Key::Cmp(*p, *a, *b)
                }
            }
            Op::Select(c, t, f) => Key::Select(*c, *t, *f),
            Op::Extract(a, n) => Key::Extract(*a, *n),
            Op::Call(f, args) => Key::Call(*f, args.clone()),
            Op::Gep(ty, p, idx) => Key::Gep(*ty, *p, idx.clone()),
            Op::Load(p) => Key::Load(*p),
            _ => return None,
        })
    }

    /// [`FOLD`], [`EXT_FOLD`], [`REUSE`], [`FACTS`] and [`LOADS`]: one walk over the
    /// dominator tree. Returns whether anything changed.
    fn merge(&mut self, passes: u32) -> bool {
        if self.blocks.is_empty() {
            return false;
        }
        let ext_fold = passes & EXT_FOLD != 0;
        let effects = self.effects();
        let tree = self.dominator_tree();
        let mut repl: Vec<Option<V>> = vec![None; self.insts.len()];
        let mut changed = false;
        // Joins of one value: until none is left (one gone can leave
        // another with one value).
        if passes & (FOLD | EXT_FOLD) != 0 {
            let mut again = true;
            while again {
                again = false;
                for &b in &tree.order {
                    for &p in &self.blocks[b].phis {
                        if repl[p as usize].is_some() {
                            continue;
                        }
                        let Op::Phi(incoming) = &self.insts[p as usize].op else {
                            continue;
                        };
                        let mut one = None;
                        let mut same = true;
                        for &(v, _) in incoming {
                            let v = resolve(&repl, v);
                            if v == V::Inst(p) {
                                continue;
                            }
                            match one {
                                None => one = Some(v),
                                Some(o) if o == v => {}
                                Some(_) => same = false,
                            }
                        }
                        if let (true, Some(v)) = (same, one) {
                            repl[p as usize] = Some(v);
                            again = true;
                            changed = true;
                        }
                    }
                }
            }
        }
        // The values every path to the block being walked has computed
        // (those of the blocks above it in the tree, and its own so far).
        let mut known: HashMap<Key, V> = HashMap::new();
        // The reads of the block being walked since its last store.
        let mut read: HashMap<Key, V> = HashMap::new();
        let mut facts = Facts::default();
        let (yes, no) = (self.bool(true), self.bool(false));
        enum Step {
            Enter(usize),
            Leave(Vec<Key>, Learned),
        }
        let mut steps = vec![Step::Enter(0)];
        while let Some(step) = steps.pop() {
            let b = match step {
                Step::Enter(b) => b,
                Step::Leave(keys, learned) => {
                    for k in keys {
                        known.remove(&k);
                    }
                    for v in learned.bits {
                        facts.bits.remove(&v);
                    }
                    for k in learned.compares {
                        facts.compares.remove(&k);
                    }
                    continue;
                }
            };
            let mut added = Vec::new();
            // Reached from one block alone, by one side of its branch: the
            // condition is that side's, here and in every block below.
            let mut learned = Learned::default();
            if passes & FACTS != 0
                && let [p] = tree.preds[b][..]
                && let Some(Term::CondBr(c, t, f)) = self.blocks[p].term
                && t != f
            {
                self.learn(
                    &repl,
                    resolve(&repl, c),
                    b == t,
                    &mut facts,
                    &mut learned,
                    8,
                    ext_fold,
                );
            }
            let known_bit = |facts: &Facts, v: V| match facts.bits.get(&v) {
                Some(true) => yes,
                Some(false) => no,
                None => v,
            };
            read.clear();
            // Two joins of the same values from the same blocks are one.
            if passes & REUSE != 0 && self.blocks[b].phis.len() > 1 {
                let mut joins: HashMap<Vec<(usize, V)>, V> = HashMap::new();
                for &p in &self.blocks[b].phis {
                    if repl[p as usize].is_some() {
                        continue;
                    }
                    let ty = self.insts[p as usize].ty;
                    let Op::Phi(incoming) = &self.insts[p as usize].op else {
                        continue;
                    };
                    let mut key: Vec<(usize, V)> = incoming
                        .iter()
                        .map(|&(v, from)| (from, resolve(&repl, v)))
                        .collect();
                    key.sort();
                    // The type too: an undefined value of one type is not
                    // that of another.
                    key.push((ty.0 as usize, V::Global(u32::MAX)));
                    match joins.get(&key) {
                        Some(&first) => {
                            repl[p as usize] = Some(first);
                            changed = true;
                        }
                        None => {
                            joins.insert(key, V::Inst(p));
                        }
                    }
                }
            }
            let body = std::mem::take(&mut self.blocks[b].body);
            let mut kept = Vec::with_capacity(body.len());
            for i in body {
                map_operands(&mut self.insts[i as usize].op, |v| {
                    let v = resolve(&repl, v);
                    let bit = known_bit(&facts, v);
                    changed |= bit != v;
                    bit
                });
                if let Op::Cmp(p, x, y) = self.insts[i as usize].op {
                    let lookup = if ext_fold {
                        canonical_cmp(p, x, y)
                    } else {
                        (p, x, y)
                    };
                    if let Some(&holds) = facts.compares.get(&lookup) {
                        repl[i as usize] = Some(if holds { yes } else { no });
                        changed = true;
                        continue;
                    }
                }
                if passes & (FOLD | EXT_FOLD) != 0
                    && let Some(v) = self.simplify(i, ext_fold, &mut changed)
                {
                    repl[i as usize] = Some(v);
                    changed = true;
                    continue;
                }
                if ext_fold
                    && passes & FACTS != 0
                    && let Op::Cmp(p, x, y) = self.insts[i as usize].op
                    && let Some(&holds) = facts.compares.get(&canonical_cmp(p, x, y))
                {
                    repl[i as usize] = Some(if holds { yes } else { no });
                    changed = true;
                    continue;
                }
                let class = self.class(&effects, i);
                if passes & REUSE != 0
                    && class == Effect::Pure
                    && let Some(key) = self.key(i, ext_fold)
                {
                    if let Some(&first) = known.get(&key) {
                        repl[i as usize] = Some(first);
                        changed = true;
                        continue;
                    }
                    known.insert(key.clone(), V::Inst(i));
                    added.push(key);
                }
                // Reads of memory, and wave operations (the same lanes run
                // the whole block): the block's own earlier one.
                if passes & LOADS != 0
                    && matches!(class, Effect::Reads | Effect::Lanes)
                    && let Some(key) = self.key(i, ext_fold)
                {
                    if let Some(&first) = read.get(&key) {
                        repl[i as usize] = Some(first);
                        changed = true;
                        continue;
                    }
                    read.insert(key, V::Inst(i));
                }
                if class == Effect::Writes {
                    read.clear();
                }
                kept.push(i);
            }
            self.blocks[b].body = kept;
            if let Some(Term::CondBr(c, t, f)) = self.blocks[b].term {
                let c = known_bit(&facts, resolve(&repl, c));
                self.blocks[b].term = Some(Term::CondBr(c, t, f));
            }
            steps.push(Step::Leave(added, learned));
            for &c in tree.children[b].iter().rev() {
                steps.push(Step::Enter(c));
            }
        }
        if !changed {
            return false;
        }
        // Everything that read a replaced value reads its replacement
        // (joins read values from blocks walked after them).
        for blk in &mut self.blocks {
            blk.phis.retain(|&p| repl[p as usize].is_none());
            if let Some(Term::CondBr(c, t, f)) = blk.term {
                blk.term = Some(Term::CondBr(resolve(&repl, c), t, f));
            }
        }
        let (blocks, insts) = (&self.blocks, &mut self.insts);
        for blk in blocks {
            for &i in blk.phis.iter().chain(&blk.body) {
                map_operands(&mut insts[i as usize].op, |v| resolve(&repl, v));
            }
        }
        true
    }

    /// [`FOLD`]: a branch on a constant goes one way. The side not taken
    /// loses the branching block's values at its joins, and so do the joins
    /// of blocks that nothing reaches any more. Returns whether a branch
    /// changed.
    fn prune(&mut self) -> bool {
        let mut changed = false;
        for b in 0..self.blocks.len() {
            let Some(Term::CondBr(c, t, f)) = self.blocks[b].term else {
                continue;
            };
            // Both sides one block: two edges into it, left as they are.
            if t == f {
                continue;
            }
            let Some((_, k)) = self.int_k(c) else {
                continue;
            };
            let (taken, not) = if k != 0 { (t, f) } else { (f, t) };
            self.blocks[b].term = Some(Term::Br(taken));
            changed = true;
            for n in 0..self.blocks[not].phis.len() {
                let p = self.blocks[not].phis[n];
                if let Op::Phi(incoming) = &mut self.insts[p as usize].op {
                    incoming.retain(|&(_, from)| from != b);
                }
            }
        }
        if !changed {
            return false;
        }
        let reached = self.reached();
        let (blocks, insts) = (&self.blocks, &mut self.insts);
        for (_, blk) in blocks.iter().enumerate().filter(|(b, _)| reached[*b]) {
            for &p in &blk.phis {
                if let Op::Phi(incoming) = &mut insts[p as usize].op {
                    incoming.retain(|&(_, from)| reached[from]);
                }
            }
        }
        true
    }

    /// Whether instruction `i` must stay whatever reads it.
    fn has_effect(&self, effects: &[Effect], i: u32) -> bool {
        match &self.insts[i as usize].op {
            Op::Store(..) | Op::AtomicRmw(..) | Op::CmpXchg(..) => true,
            Op::Call(f, _) => effects[*f as usize] == Effect::Writes,
            _ => false,
        }
    }

    /// [`DEAD`]: keeps the instructions with an effect, the branch
    /// conditions, and what they read, transitively; the rest leave their
    /// blocks. Blocks no branch reaches are emptied (they are not written).
    fn drop_dead(&mut self) {
        let effects = self.effects();
        let reached = self.reached();
        let mut live = vec![false; self.insts.len()];
        let mut work: Vec<u32> = Vec::new();
        let mark = |live: &mut [bool], work: &mut Vec<u32>, v: V| {
            if let V::Inst(i) = v
                && !live[i as usize]
            {
                live[i as usize] = true;
                work.push(i);
            }
        };
        for (b, blk) in self.blocks.iter().enumerate() {
            if !reached[b] {
                continue;
            }
            for &i in &blk.body {
                if self.has_effect(&effects, i) {
                    mark(&mut live, &mut work, V::Inst(i));
                }
            }
            if let Some(Term::CondBr(c, ..)) = blk.term {
                mark(&mut live, &mut work, c);
            }
        }
        while let Some(i) = work.pop() {
            for_each_operand(&self.insts[i as usize].op, |v| {
                mark(&mut live, &mut work, v)
            });
        }
        for (b, blk) in self.blocks.iter_mut().enumerate() {
            if reached[b] {
                blk.body.retain(|&i| live[i as usize]);
                blk.phis.retain(|&i| live[i as usize]);
            } else {
                blk.body.clear();
                blk.phis.clear();
            }
        }
        self.drop_unused_functions();
        self.drop_unused_tables();
    }

    /// See [`Usage`]: over the blocks reached from the entry.
    pub(crate) fn usage(&self) -> Usage {
        let mut u = Usage::default();
        let reached = self.reached();
        let wide = |ty: Ty| self.types[ty.0 as usize] == TypeDef::Int(64);
        // The first field of a handle's annotation: its kind, with the UAV
        // bit; the second: the component type and count.
        let annotation = |h: V| -> Option<(i64, i64)> {
            let V::Inst(a) = h else { return None };
            let Op::Call(f, args) = &self.insts[a as usize].op else {
                return None;
            };
            if self.funcs[*f as usize].name != "dx.op.annotateHandle" {
                return None;
            }
            let V::Const(props) = *args.get(2)? else {
                return None;
            };
            let Const::Aggregate(fields) = &self.consts[props as usize].1 else {
                return None;
            };
            match (
                &self.consts[fields[0] as usize].1,
                &self.consts[fields[1] as usize].1,
            ) {
                (Const::Int(kind), Const::Int(components)) => Some((*kind, *components)),
                _ => None,
            }
        };
        for (_, blk) in self.blocks.iter().enumerate().filter(|(b, _)| reached[*b]) {
            for &i in blk.phis.iter().chain(&blk.body) {
                let inst = &self.insts[i as usize];
                u.int64 |= wide(inst.ty);
                for_each_operand(&inst.op, |v| u.int64 |= wide(self.type_of(v)));
                let Op::Call(f, args) = &inst.op else {
                    continue;
                };
                let name = self.funcs[*f as usize].name.as_str();
                let name = name.strip_prefix("dx.op.").unwrap_or(name);
                if name.starts_with("wave") || name.starts_with("quad") {
                    u.wave_ops = true;
                } else if name == "createHandleFromHeap" {
                    match args.get(2).and_then(|&s| self.int_k(s)) {
                        Some((_, 0)) => u.heap_views = true,
                        _ => u.heap_samplers = true,
                    }
                } else if name == "annotateHandle" {
                    u.rovs |= annotation(V::Inst(i)).is_some_and(|(kind, _)| kind & 0x2000 != 0);
                } else if name.starts_with("atomic") && name.ends_with(".i64") {
                    u.atomic64 = true;
                } else if name.starts_with("textureLoad") || name.starts_with("bufferLoad") {
                    u.typed_uav_loads |= args.get(1).and_then(|&h| annotation(h)).is_none_or(
                        |(kind, components)| kind & 0x1000 != 0 && components >> 8 & 0xff > 1,
                    );
                }
            }
        }
        u
    }

    /// A constant table no remaining load reads leaves the module (the
    /// validator refuses an internal declaration nothing uses). Shared
    /// memory stays declared whether it is read or not.
    fn drop_unused_tables(&mut self) {
        let mut used: Vec<bool> = self.globals.iter().map(|g| g.init.is_none()).collect();
        if used.iter().all(|&u| u) {
            return;
        }
        for blk in &self.blocks {
            for &i in blk.phis.iter().chain(&blk.body) {
                for_each_operand(&self.insts[i as usize].op, |v| {
                    if let V::Global(g) = v {
                        used[g as usize] = true;
                    }
                });
            }
        }
        for md in &self.mds {
            if let MdDef::Value(_, V::Global(g)) = md {
                used[*g as usize] = true;
            }
        }
        if used.iter().all(|&u| u) {
            return;
        }
        let mut new = vec![u32::MAX; used.len()];
        for (next, (old, _)) in used.iter().enumerate().filter(|u| *u.1).enumerate() {
            new[old] = next as u32;
        }
        let mut old = 0;
        self.globals.retain(|_| {
            old += 1;
            used[old - 1]
        });
        let renumber = |v: V| match v {
            V::Global(g) if used[g as usize] => V::Global(new[g as usize]),
            v => v,
        };
        for inst in &mut self.insts {
            map_operands(&mut inst.op, renumber);
        }
        for md in &mut self.mds {
            if let MdDef::Value(_, v) = md {
                *v = renumber(*v);
            }
        }
    }

    /// Declarations no remaining call names leave the module (the validator
    /// refuses an external function nothing uses); the others keep their
    /// order, so `main` stays function 0.
    fn drop_unused_functions(&mut self) {
        let mut used: Vec<bool> = self.funcs.iter().map(|f| f.body).collect();
        for blk in &self.blocks {
            for &i in blk.phis.iter().chain(&blk.body) {
                if let Op::Call(f, _) = self.insts[i as usize].op {
                    used[f as usize] = true;
                }
            }
        }
        for md in &self.mds {
            if let MdDef::Value(_, V::Func(f)) = md {
                used[*f as usize] = true;
            }
        }
        if used.iter().all(|&u| u) {
            return;
        }
        let mut new = vec![u32::MAX; used.len()];
        for (next, (old, _)) in used.iter().enumerate().filter(|u| *u.1).enumerate() {
            new[old] = next as u32;
        }
        let mut old = 0;
        self.funcs.retain(|_| {
            old += 1;
            used[old - 1]
        });
        self.func_ids = self
            .funcs
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name.clone(), i as u32))
            .collect();
        // Instructions that left their block keep a stale number: nothing
        // reads them again.
        for inst in &mut self.insts {
            if let Op::Call(f, _) = &mut inst.op
                && used[*f as usize]
            {
                *f = new[*f as usize];
            }
        }
        for md in &mut self.mds {
            if let MdDef::Value(_, V::Func(f)) = md {
                *f = new[*f as usize];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `main` with one block and a function of each kind of effect.
    struct Kit {
        m: Module,
        b: usize,
        pure: V,
        reads: V,
        writes: V,
    }

    fn kit() -> Kit {
        let mut m = Module::default();
        let i32t = m.int(32);
        let void = m.void();
        let fty = m.func_ty(i32t, vec![i32t, i32t]);
        m.ptr(fty);
        let pure = m.function("dx.op.binary.i32", fty, Some(Attr::ReadNone), false);
        let reads = m.function("dx.op.bufferLoad.i32", fty, Some(Attr::ReadOnly), false);
        let writes = m.function("dx.op.bufferStore.i32", fty, Some(Attr::NoUnwind), false);
        let main_ty = m.func_ty(void, vec![]);
        m.ptr(main_ty);
        m.function("main", main_ty, None, true);
        let b = m.new_block();
        Kit {
            m,
            b,
            pure,
            reads,
            writes,
        }
    }

    fn body(m: &Module, b: usize) -> usize {
        m.blocks[b].body.len()
    }

    #[test]
    fn unread_results_without_effect_are_dropped() {
        let Kit {
            mut m,
            b,
            pure,
            reads,
            writes,
        } = kit();
        let (one, two) = (m.i32(1), m.i32(2));
        let used = m.call(b, reads, vec![one, two]);
        let dead = m.call(b, reads, vec![two, one]);
        let chain = m.bin(b, Bin::Add, dead, one);
        let _unread = m.call(b, pure, vec![chain, two]);
        // A store's result is unread and stays, with what it reads.
        let kept = m.bin(b, Bin::Add, used, two);
        m.call(b, writes, vec![kept, one]);
        m.terminate(b, Term::Ret);
        m.optimise(DEAD);
        assert_eq!(body(&m, b), 3, "the load, the add and the store");
        m.write();
    }

    #[test]
    fn branch_conditions_and_phi_cycles() {
        let Kit {
            mut m,
            b,
            reads,
            writes,
            ..
        } = kit();
        let i32t = m.int(32);
        let (zero, one) = (m.i32(0), m.i32(1));
        let (head, exit) = (m.new_block(), m.new_block());
        m.terminate(b, Term::Br(head));
        // A counter that decides the branch, and a register that only
        // feeds itself round the loop.
        let count = m.phi(head, i32t);
        let idle = m.phi(head, i32t);
        let next = m.bin(head, Bin::Add, count, one);
        let idle_next = m.call(head, reads, vec![idle, one]);
        let more = m.cmp(head, pred::IULT, next, one);
        m.terminate(head, Term::CondBr(more, head, exit));
        m.set_phi(count, vec![(zero, b), (next, head)]);
        m.set_phi(idle, vec![(zero, b), (idle_next, head)]);
        m.call(exit, writes, vec![zero, one]);
        m.terminate(exit, Term::Ret);
        m.optimise(DEAD);
        assert_eq!(m.blocks[head].phis.len(), 1, "the counter");
        assert_eq!(body(&m, head), 2, "the step and the test");
        assert_eq!(body(&m, exit), 1);
        m.write();
    }

    #[test]
    fn wave_operations_have_no_effect_of_their_own() {
        let mut k = kit();
        let i32t = k.m.int(32);
        let fty = k.m.func_ty(i32t, vec![i32t, i32t]);
        let first = k.m.function(
            "dx.op.waveReadLaneFirst.i32",
            fty,
            Some(Attr::NoUnwind),
            false,
        );
        let barrier =
            k.m.function("dx.op.barrier", fty, Some(Attr::NoDuplicate), false);
        let one = k.m.i32(1);
        k.m.call(k.b, first, vec![one, one]);
        k.m.call(k.b, barrier, vec![one, one]);
        k.m.terminate(k.b, Term::Ret);
        k.m.optimise(DEAD);
        assert_eq!(body(&k.m, k.b), 1, "the barrier");
        // The declarations nothing calls any more are gone, and the call
        // that stayed names its function by the new number.
        let names: Vec<&str> = k.m.funcs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["main", "dx.op.barrier"]);
        let Op::Call(f, _) = k.m.insts[k.m.blocks[k.b].body[0] as usize].op else {
            panic!("a call");
        };
        assert_eq!(k.m.funcs[f as usize].name, "dx.op.barrier");
        k.m.write();
    }

    /// The operands of each store left in block `b`.
    fn stores(m: &Module, b: usize) -> Vec<Vec<V>> {
        m.blocks[b]
            .body
            .iter()
            .filter_map(|&i| match &m.insts[i as usize].op {
                Op::Call(g, args) if m.funcs[*g as usize].name.contains("Store") => {
                    Some(args.clone())
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn constants_and_identities_fold() {
        let Kit {
            mut m,
            b,
            reads,
            writes,
            ..
        } = kit();
        let (i32t, f32t) = (m.int(32), m.float());
        let (zero, one, all) = (m.i32(0), m.i32(1), m.i32(u32::MAX));
        let x = m.call(b, reads, vec![one, one]);
        // x + 0, -1 & x, x >> 0, x * 1, x | x: all x.
        let a = m.bin(b, Bin::Add, x, zero);
        let a = m.bin(b, Bin::And, all, a);
        let a = m.bin(b, Bin::LShr, a, zero);
        let a = m.bin(b, Bin::Mul, a, one);
        let a = m.bin(b, Bin::Or, a, a);
        // There and back.
        let f = m.cast(b, Cast::Bitcast, a, f32t);
        let a = m.cast(b, Cast::Bitcast, f, i32t);
        // Constants: (262168 >> 16) & 127 = 4, 7 & 31 = 7, -8 >>s 1 = -4.
        let k = m.i32(262168);
        let sixteen = m.i32(16);
        let k = m.bin(b, Bin::LShr, k, sixteen);
        let k127 = m.i32(127);
        let k = m.bin(b, Bin::And, k, k127);
        let minus8 = m.i32(-8i32 as u32);
        let s = m.bin(b, Bin::AShr, minus8, one);
        // A shift by the width is left alone.
        let k32 = m.i32(32);
        let wide = m.bin(b, Bin::Shl, one, k32);
        // A select between a value and itself, and on a constant.
        let c = m.cmp(b, pred::IULT, zero, one);
        let sel = m.select(b, c, a, wide);
        let same = m.select(b, c, sel, sel);
        m.call(b, writes, vec![same, k]);
        m.call(b, writes, vec![s, wide]);
        m.terminate(b, Term::Ret);
        m.optimise(FOLD | DEAD);
        assert_eq!(body(&m, b), 4, "the read, the wide shift and two stores");
        let four = m.i32(4);
        let minus4 = m.i32(-4i32 as u32);
        let stores = stores(&m, b);
        assert_eq!(stores, [vec![x, four], vec![minus4, wide]]);
        m.write();
    }

    #[test]
    fn float_arithmetic_is_not_folded() {
        let Kit {
            mut m, b, writes, ..
        } = kit();
        let f32t = m.float();
        let fty = m.func_ty(f32t, vec![f32t, f32t]);
        let input = m.function("dx.op.loadInput.f32", fty, Some(Attr::ReadNone), false);
        let (zero, one, two) = (m.f32(0), m.f32(1.0f32.to_bits()), m.f32(2.0f32.to_bits()));
        let x = m.call(b, input, vec![zero, zero]);
        let y = m.call(b, input, vec![one, one]);
        let a = m.bin(b, Bin::Add, x, zero);
        let p = m.bin(b, Bin::Mul, a, one);
        // The same sum the other way round is another instruction.
        let xy = m.bin(b, Bin::Add, x, y);
        let yx = m.bin(b, Bin::Add, y, x);
        // Constants compare here: 1 < 2, and nothing equals a NaN.
        let nan = m.f32(f32::NAN.to_bits());
        let lt = m.cmp(b, pred::OLT, one, two);
        let eq = m.cmp(b, pred::OEQ, nan, nan);
        let une = m.cmp(b, pred::UNE, nan, one);
        let sel = m.select(b, lt, xy, yx);
        let sel = m.select(b, eq, p, sel);
        let sel = m.select(b, une, sel, p);
        m.call(b, writes, vec![sel, p]);
        m.terminate(b, Term::Ret);
        m.optimise(ALL);
        // Two inputs, x + 0.0, * 1.0, x + y and the store: y + x and the
        // selects are gone.
        assert_eq!(body(&m, b), 6);
        assert_eq!(stores(&m, b), [[xy, p]]);
        m.write();
    }

    /// entry: `c = x < y`, branch to `then` or `other`, which join.
    struct Diamond {
        k: Kit,
        x: V,
        y: V,
        c: V,
        then: usize,
        other: usize,
        join: usize,
    }

    fn diamond() -> Diamond {
        let mut k = kit();
        let (one, two) = (k.m.i32(1), k.m.i32(2));
        let x = k.m.call(k.b, k.reads, vec![one, one]);
        let y = k.m.call(k.b, k.reads, vec![two, two]);
        let c = k.m.cmp(k.b, pred::IULT, x, y);
        let (then, other, join) = (k.m.new_block(), k.m.new_block(), k.m.new_block());
        k.m.terminate(k.b, Term::CondBr(c, then, other));
        k.m.terminate(then, Term::Br(join));
        k.m.terminate(other, Term::Br(join));
        k.m.terminate(join, Term::Ret);
        Diamond {
            k,
            x,
            y,
            c,
            then,
            other,
            join,
        }
    }

    #[test]
    fn a_value_is_reused_below_its_block_and_not_beside_it() {
        let Diamond {
            k:
                Kit {
                    mut m,
                    b,
                    pure,
                    writes,
                    ..
                },
            x,
            y,
            then,
            other,
            join,
            ..
        } = diamond();
        // The entry's value again in a branch; a branch's value again in
        // the other branch and in the join (neither is on every path).
        let top = m.call(b, pure, vec![x, y]);
        m.call(b, writes, vec![top, top]);
        let again = m.call(then, pure, vec![x, y]);
        let side = m.bin(then, Bin::Add, y, x);
        m.call(then, writes, vec![again, side]);
        let beside = m.bin(other, Bin::Add, x, y);
        m.call(other, writes, vec![beside, beside]);
        let after = m.bin(join, Bin::Add, x, y);
        m.call(join, writes, vec![after, after]);
        m.optimise(REUSE);
        assert_eq!(stores(&m, then), [[top, side]]);
        assert_eq!(body(&m, then), 2);
        assert_eq!(body(&m, other), 2);
        assert_eq!(body(&m, join), 2);
        m.write();
    }

    #[test]
    fn a_branch_taken_says_what_its_condition_is() {
        let Diamond {
            k:
                Kit {
                    mut m,
                    b,
                    reads,
                    writes,
                    ..
                },
            x,
            y,
            c,
            then,
            other,
            join,
        } = diamond();
        let three = m.i32(3);
        let d = m.call(b, reads, vec![three, three]);
        let d = m.cmp(b, pred::NE, d, three);
        let both = m.bin(b, Bin::And, c, d);
        // Behind the branch: the condition again (a guard), its opposite
        // comparison, and a guarded write.
        let guard = m.bin(then, Bin::And, c, d);
        let old = m.select(then, guard, x, y);
        let not = m.cmp(then, pred::IUGE, x, y);
        let new = m.select(then, not, y, old);
        m.call(then, writes, vec![new, new]);
        let kept = m.select(other, c, x, y);
        m.call(other, writes, vec![kept, kept]);
        // After the join nothing is known.
        let open = m.select(join, c, x, y);
        m.call(join, writes, vec![open, open]);
        m.optimise(FOLD | FACTS);
        // then: select(d, x, y) and the store; other: y is stored.
        assert_eq!(body(&m, then), 2);
        let V::Inst(sel) = stores(&m, then)[0][0] else {
            panic!()
        };
        assert!(matches!(m.insts[sel as usize].op, Op::Select(g, a, z) if (g, a, z) == (d, x, y)));
        assert_eq!(stores(&m, other), [[y, y]]);
        assert_eq!(body(&m, join), 2);
        let _ = both;
        m.write();

        // Both terms of an `and` that held hold behind it.
        let Diamond {
            k:
                Kit {
                    mut m,
                    b,
                    reads,
                    writes,
                    ..
                },
            x,
            y,
            c,
            ..
        } = diamond();
        let three = m.i32(3);
        let d = m.call(b, reads, vec![three, three]);
        let d = m.cmp(b, pred::NE, d, three);
        let both = m.bin(b, Bin::And, c, d);
        let (inner, out) = (m.new_block(), m.new_block());
        m.terminate(b, Term::CondBr(both, inner, out));
        let a = m.select(inner, c, x, y);
        let z = m.select(inner, d, a, y);
        m.call(inner, writes, vec![z, z]);
        m.terminate(inner, Term::Br(out));
        // Where the `and` did not hold either term may.
        let w = m.select(out, d, x, y);
        m.call(out, writes, vec![w, w]);
        m.terminate(out, Term::Ret);
        m.optimise(FOLD | FACTS);
        assert_eq!(stores(&m, inner), [[x, x]]);
        assert_eq!(body(&m, out), 2);
        m.write();
    }

    #[test]
    fn a_branch_on_a_constant_goes_one_way() {
        let Diamond {
            k: Kit {
                mut m, b, writes, ..
            },
            x,
            y,
            c,
            then,
            other,
            join,
        } = diamond();
        let i32t = m.int(32);
        // Behind the branch the same test again: its other side is never
        // run, and the join behind it has one value.
        let (deep, never, meet) = (m.new_block(), m.new_block(), m.new_block());
        m.terminate(then, Term::CondBr(c, deep, never));
        m.terminate(deep, Term::Br(meet));
        m.call(never, writes, vec![y, y]);
        m.terminate(never, Term::Br(meet));
        let p = m.phi(meet, i32t);
        m.set_phi(p, vec![(x, deep), (y, never)]);
        m.call(meet, writes, vec![p, p]);
        m.terminate(meet, Term::Br(join));
        let q = m.phi(join, i32t);
        m.set_phi(q, vec![(p, meet), (y, other)]);
        m.call(join, writes, vec![q, q]);
        m.optimise(ALL);
        assert!(matches!(m.blocks[then].term, Some(Term::Br(t)) if t == deep));
        assert_eq!(stores(&m, meet), [[x, x]]);
        assert!(m.blocks[meet].phis.is_empty());
        assert_eq!(body(&m, never), 0, "emptied: nothing reaches it");
        let Op::Phi(incoming) = &m.insts[m.blocks[join].phis[0] as usize].op else {
            panic!()
        };
        assert_eq!(incoming, &vec![(x, meet), (y, other)]);
        let _ = b;
        m.write();
    }

    #[test]
    fn a_read_is_reused_within_its_block_until_a_store() {
        let Diamond {
            k:
                Kit {
                    mut m,
                    b,
                    reads,
                    writes,
                    ..
                },
            x,
            y,
            then,
            ..
        } = diamond();
        let i32t = m.int(32);
        // The entry reads (1, 1) for x: again in the entry it is x; in the
        // next block it is read again.
        let one = m.i32(1);
        let again = m.call(b, reads, vec![one, one]);
        m.call(b, writes, vec![again, y]);
        let after = m.call(b, reads, vec![one, one]);
        let twice = m.call(b, reads, vec![one, one]);
        let sum = m.bin(b, Bin::Add, after, twice);
        m.call(b, writes, vec![sum, sum]);
        let below = m.call(then, reads, vec![one, one]);
        m.call(then, writes, vec![below, below]);
        // A constant buffer cannot change: one load for the function.
        let cfty = m.func_ty(i32t, vec![i32t, i32t]);
        let cb = m.function(
            "dx.op.cbufferLoadLegacy.i32",
            cfty,
            Some(Attr::ReadOnly),
            false,
        );
        let top = m.call(b, cb, vec![one, one]);
        m.call(b, writes, vec![top, top]);
        let under = m.call(then, cb, vec![one, one]);
        m.call(then, writes, vec![under, under]);
        m.optimise(REUSE | LOADS);
        let stores = stores(&m, b);
        assert_eq!(stores[0], [x, y], "the second read is the first");
        let V::Inst(sum) = stores[1][0] else { panic!() };
        assert!(
            matches!(m.insts[sum as usize].op, Op::Bin(Bin::Add, a, z) if a == after && z == after),
            "read again after the store, once"
        );
        assert_eq!(stores[2], [top, top]);
        assert_eq!(
            super::tests::stores(&m, then),
            [vec![below, below], vec![top, top]]
        );
        m.write();
    }

    #[test]
    fn a_join_of_one_value_and_two_equal_joins() {
        let Diamond {
            k: Kit { mut m, writes, .. },
            x,
            y,
            then,
            other,
            join,
            ..
        } = diamond();
        let i32t = m.int(32);
        let one = m.phi(join, i32t);
        m.set_phi(one, vec![(x, then), (x, other)]);
        let p = m.phi(join, i32t);
        m.set_phi(p, vec![(x, then), (y, other)]);
        let q = m.phi(join, i32t);
        m.set_phi(q, vec![(y, other), (x, then)]);
        m.call(join, writes, vec![one, p]);
        let sum = m.bin(join, Bin::Add, p, q);
        m.call(join, writes, vec![sum, q]);
        m.optimise(FOLD | REUSE);
        assert_eq!(
            m.blocks[join].phis,
            [match p {
                V::Inst(i) => i,
                _ => unreachable!(),
            }]
        );
        let stores = stores(&m, join);
        assert_eq!(stores[0], [x, p]);
        assert_eq!(stores[1][1], p);
        m.write();
    }

    // ---- the passes against an interpreter, on random programs ----

    /// Runs `m` on `inputs`: the stores it makes, in order (and how it
    /// ended). Reads a value before its definition ran: a panic.
    fn run(m: &Module, inputs: &[u32]) -> Vec<(String, Vec<u64>)> {
        let mut vals: Vec<Option<u64>> = vec![None; m.insts.len()];
        let mut memory = [0u64; 16];
        let mut trace = Vec::new();
        let eval = |vals: &[Option<u64>], v: V| -> u64 {
            match v {
                V::Inst(i) => vals[i as usize].expect("a value defined before its use"),
                V::Const(c) => {
                    match (&m.consts[c as usize].1, m.int_bits(m.consts[c as usize].0)) {
                        (Const::Int(x), Some(bits)) => *x as u64 & mask(bits),
                        (Const::Float(bits), _) => u64::from(*bits),
                        _ => 0,
                    }
                }
                _ => panic!("a global or function as a value"),
            }
        };
        let (mut b, mut from) = (0usize, usize::MAX);
        for _ in 0..4000 {
            let blk = &m.blocks[b];
            // Joins take their values together, before any is assigned.
            let joined: Vec<(u32, u64)> = blk
                .phis
                .iter()
                .map(|&p| {
                    let Op::Phi(incoming) = &m.insts[p as usize].op else {
                        panic!("a join")
                    };
                    let v = incoming
                        .iter()
                        .find(|x| x.1 == from)
                        .expect("a value for the edge taken");
                    (p, eval(&vals, v.0))
                })
                .collect();
            for (p, x) in joined {
                vals[p as usize] = Some(x);
            }
            for &i in &blk.body {
                let inst = &m.insts[i as usize];
                let bits = m.int_bits(inst.ty);
                let float = |x: u64| f32::from_bits(x as u32);
                let r = match &inst.op {
                    Op::Bin(op, x, y) => {
                        let (x, y) = (eval(&vals, *x), eval(&vals, *y));
                        match bits {
                            Some(bits) => {
                                let r = match op {
                                    Bin::Add => x.wrapping_add(y),
                                    Bin::Sub => x.wrapping_sub(y),
                                    Bin::Mul => x.wrapping_mul(y),
                                    Bin::And => x & y,
                                    Bin::Or => x | y,
                                    Bin::Xor => x ^ y,
                                    Bin::Shl => x << y,
                                    Bin::LShr => x >> y,
                                    Bin::AShr => (signed(bits, x) >> y) as u64,
                                    _ => panic!("no divisions here"),
                                };
                                r & mask(bits)
                            }
                            None => {
                                let (x, y) = (float(x), float(y));
                                let r = match op {
                                    Bin::Add => x + y,
                                    Bin::Sub => x - y,
                                    _ => x * y,
                                };
                                u64::from(r.to_bits())
                            }
                        }
                    }
                    Op::Cmp(p, x, y) => {
                        let wide = m.int_bits(m.type_of(*x));
                        let (x, y) = (eval(&vals, *x), eval(&vals, *y));
                        let holds = match wide {
                            Some(bits) => {
                                let (sx, sy) = (signed(bits, x), signed(bits, y));
                                match *p {
                                    pred::EQ => x == y,
                                    pred::NE => x != y,
                                    pred::IUGT => x > y,
                                    pred::IUGE => x >= y,
                                    pred::IULT => x < y,
                                    pred::IULE => x <= y,
                                    pred::SGT => sx > sy,
                                    pred::SGE => sx >= sy,
                                    pred::SLT => sx < sy,
                                    _ => sx <= sy,
                                }
                            }
                            None => {
                                let (x, y) = (float(x), float(y));
                                if x.is_nan() || y.is_nan() {
                                    p & 8 != 0
                                } else {
                                    (p & 1 != 0 && x == y)
                                        || (p & 2 != 0 && x > y)
                                        || (p & 4 != 0 && x < y)
                                }
                            }
                        };
                        u64::from(holds)
                    }
                    Op::Select(c, t, f) => {
                        if eval(&vals, *c) != 0 {
                            eval(&vals, *t)
                        } else {
                            eval(&vals, *f)
                        }
                    }
                    Op::Cast(op, x) => {
                        let from_bits = m.int_bits(m.type_of(*x));
                        let x = eval(&vals, *x);
                        match (op, from_bits, bits) {
                            (Cast::Bitcast, _, _) => x,
                            (Cast::ZExt, _, _) => x,
                            (Cast::SExt, Some(f), Some(t)) => signed(f, x) as u64 & mask(t),
                            (Cast::Trunc, _, Some(t)) => x & mask(t),
                            _ => panic!("a cast the generator does not make"),
                        }
                    }
                    Op::Call(f, args) => {
                        let a: Vec<u64> = args.iter().map(|&v| eval(&vals, v)).collect();
                        let name = m.funcs[*f as usize].name.as_str();
                        match name {
                            "dx.op.loadInput.i32" => {
                                u64::from(inputs[(a[0] ^ a[1]) as usize % inputs.len()])
                            }
                            "dx.op.binary.i32" => a[0].min(a[1]),
                            "dx.op.cbufferLoadLegacy.i32" => u64::from(
                                inputs[a[0] as usize % inputs.len()].rotate_left(a[1] as u32 & 31),
                            ),
                            "dx.op.bufferLoad.i32" => memory[(a[0] ^ a[1]) as usize % 16],
                            "dx.op.waveReadLaneFirst.i32" => a[0] ^ a[1],
                            _ => {
                                memory[a[0] as usize % 16] = a[1];
                                trace.push((name.to_owned(), a));
                                0
                            }
                        }
                    }
                    _ => panic!("an instruction the generator does not make"),
                };
                vals[i as usize] = Some(r);
            }
            from = b;
            b = match blk.term.expect("a terminator") {
                Term::Ret => {
                    trace.push(("ret".into(), Vec::new()));
                    return trace;
                }
                Term::Br(t) => t,
                Term::CondBr(c, t, f) => {
                    if eval(&vals, c) != 0 {
                        t
                    } else {
                        f
                    }
                }
            };
        }
        trace.push(("still running".into(), Vec::new()));
        trace
    }

    /// A random structured program over a few registers, as the lowering
    /// makes them: every register joined at every join, conditions tested
    /// again behind their branch, values computed again.
    struct Gen {
        k: Kit,
        rng: u64,
        regs: Vec<V>,
        conds: Vec<V>,
        cur: usize,
        input: V,
        table: V,
        wave: V,
        /// The operands of the block's last read (to read the place again,
        /// and to store to it).
        last: Option<(V, V)>,
    }

    impl Gen {
        fn next(&mut self, n: u64) -> u64 {
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 7;
            self.rng ^= self.rng << 17;
            (self.rng >> 11) % n
        }

        fn reg(&mut self) -> V {
            let n = self.next(self.regs.len() as u64) as usize;
            self.regs[n]
        }

        /// A register or a small constant.
        fn value(&mut self) -> V {
            match self.next(4) {
                0 => {
                    let k = [0u32, 1, 2, 31, u32::MAX, 0x8000_0000, 7][self.next(7) as usize];
                    self.k.m.i32(k)
                }
                _ => self.reg(),
            }
        }

        fn cond(&mut self) -> V {
            let c = self.cur;
            let made = match self.next(6) {
                0 if !self.conds.is_empty() => {
                    let n = self.next(self.conds.len() as u64) as usize;
                    return self.conds[n];
                }
                1 if self.conds.len() > 1 => {
                    let (a, b) = (
                        self.next(self.conds.len() as u64),
                        self.next(self.conds.len() as u64),
                    );
                    let op = [Bin::And, Bin::Or, Bin::Xor][self.next(3) as usize];
                    self.k
                        .m
                        .bin(c, op, self.conds[a as usize], self.conds[b as usize])
                }
                2 if !self.conds.is_empty() => {
                    let n = self.next(self.conds.len() as u64) as usize;
                    let t = self.k.m.bool(true);
                    self.k.m.bin(c, Bin::Xor, self.conds[n], t)
                }
                _ => {
                    let p = pred::EQ + self.next(10) as u8;
                    let (x, y) = (self.reg(), self.value());
                    self.k.m.cmp(c, p, x, y)
                }
            };
            self.conds.push(made);
            made
        }

        fn statement(&mut self) {
            let c = self.cur;
            let d = self.next(self.regs.len() as u64) as usize;
            let i32t = self.k.m.int(32);
            let f32t = self.k.m.float();
            let r = match self.next(14) {
                0..=3 => {
                    let op = [Bin::Add, Bin::Sub, Bin::Mul, Bin::And, Bin::Or, Bin::Xor]
                        [self.next(6) as usize];
                    let (x, y) = (self.value(), self.value());
                    self.k.m.bin(c, op, x, y)
                }
                4 => {
                    let op = [Bin::Shl, Bin::LShr, Bin::AShr][self.next(3) as usize];
                    let x = self.value();
                    let by = self.next(32) as u32;
                    let by = self.k.m.i32(by);
                    self.k.m.bin(c, op, x, by)
                }
                5 | 6 => {
                    // A guarded write: the old value where the guard fails.
                    let g = self.cond();
                    let x = self.value();
                    self.k.m.select(self.cur, g, x, self.regs[d])
                }
                7 => {
                    let (x, y) = (self.value(), self.value());
                    let f = [self.k.pure, self.input, self.table, self.wave][self.next(4) as usize];
                    self.k.m.call(c, f, vec![x, y])
                }
                8 | 9 => {
                    let again = self.next(2) == 0;
                    let (x, y) = match self.last {
                        Some(last) if again => last,
                        _ => (self.value(), self.value()),
                    };
                    self.last = Some((x, y));
                    self.k.m.call(c, self.k.reads, vec![x, y])
                }
                10 => {
                    // A store, half the time to the place read last.
                    let (mut x, y) = (self.value(), self.value());
                    let there = self.next(2) == 0;
                    if let Some((a, b)) = self.last
                        && there
                    {
                        x = self.k.m.bin(c, Bin::Xor, a, b);
                    }
                    self.k.m.call(c, self.k.writes, vec![x, y]);
                    return;
                }
                11 => {
                    // Float arithmetic, read back as bits.
                    let (x, y) = (self.value(), self.value());
                    let (x, y) = (
                        self.k.m.cast(c, Cast::Bitcast, x, f32t),
                        self.k.m.cast(c, Cast::Bitcast, y, f32t),
                    );
                    let op = [Bin::Add, Bin::Sub, Bin::Mul][self.next(3) as usize];
                    let f = self.k.m.bin(c, op, x, y);
                    if self.next(2) == 0 {
                        let p = self.next(16) as u8;
                        let t = self.k.m.cmp(c, p, f, y);
                        self.conds.push(t);
                    }
                    self.k.m.cast(c, Cast::Bitcast, f, i32t)
                }
                12 => {
                    let g = self.cond();
                    self.k.m.cast(self.cur, Cast::ZExt, g, i32t)
                }
                _ => {
                    let x = self.value();
                    let f = self.k.m.cast(c, Cast::Bitcast, x, f32t);
                    self.k.m.cast(c, Cast::Bitcast, f, i32t)
                }
            };
            self.regs[d] = r;
        }

        fn code(&mut self, depth: u32) {
            for _ in 0..1 + self.next(6) {
                match self.next(8) {
                    0 | 1 if depth > 0 => self.branch(depth - 1),
                    2 if depth > 0 => self.repeat(depth - 1),
                    _ => self.statement(),
                }
            }
        }

        /// if / else, every register joined.
        fn branch(&mut self, depth: u32) {
            let c = self.cond();
            let i32t = self.k.m.int(32);
            let (then, other, join) = (
                self.k.m.new_block(),
                self.k.m.new_block(),
                self.k.m.new_block(),
            );
            self.k.m.terminate(self.cur, Term::CondBr(c, then, other));
            let (before, known) = (self.regs.clone(), self.conds.len());
            self.cur = then;
            self.last = None;
            self.code(depth);
            let (then_regs, then_end) = (std::mem::replace(&mut self.regs, before), self.cur);
            self.k.m.terminate(then_end, Term::Br(join));
            self.conds.truncate(known);
            self.cur = other;
            self.last = None;
            if self.next(3) != 0 {
                self.code(depth);
            }
            let other_end = self.cur;
            self.k.m.terminate(other_end, Term::Br(join));
            self.conds.truncate(known);
            for (r, &t) in then_regs.iter().enumerate() {
                let p = self.k.m.phi(join, i32t);
                self.k
                    .m
                    .set_phi(p, vec![(t, then_end), (self.regs[r], other_end)]);
                self.regs[r] = p;
            }
            self.cur = join;
            self.last = None;
        }

        /// A loop that runs at most a few times, every register joined at
        /// its head.
        fn repeat(&mut self, depth: u32) {
            let i32t = self.k.m.int(32);
            let (head, exit) = (self.k.m.new_block(), self.k.m.new_block());
            let before = self.cur;
            self.k.m.terminate(before, Term::Br(head));
            let entering = self.regs.clone();
            let joins: Vec<V> = (0..self.regs.len())
                .map(|_| self.k.m.phi(head, i32t))
                .collect();
            let count = self.k.m.phi(head, i32t);
            self.regs = joins.clone();
            let known = self.conds.len();
            self.cur = head;
            self.last = None;
            self.code(depth);
            let one = self.k.m.i32(1);
            let next = self.k.m.bin(self.cur, Bin::Add, count, one);
            let limit = 1 + self.next(3) as u32;
            let limit = self.k.m.i32(limit);
            let mut more = self.k.m.cmp(self.cur, pred::IULT, next, limit);
            if self.next(2) == 0 {
                let c = self.cond();
                more = self.k.m.bin(self.cur, Bin::And, more, c);
            }
            let latch = self.cur;
            self.k.m.terminate(latch, Term::CondBr(more, head, exit));
            for (r, &p) in joins.iter().enumerate() {
                self.k
                    .m
                    .set_phi(p, vec![(entering[r], before), (self.regs[r], latch)]);
            }
            let zero = self.k.m.i32(0);
            self.k.m.set_phi(count, vec![(zero, before), (next, latch)]);
            self.conds.truncate(known);
            self.cur = exit;
            self.last = None;
        }
    }

    fn random_program(seed: u64) -> Module {
        let mut k = kit();
        let i32t = k.m.int(32);
        let fty = k.m.func_ty(i32t, vec![i32t, i32t]);
        let input =
            k.m.function("dx.op.loadInput.i32", fty, Some(Attr::ReadNone), false);
        let table = k.m.function(
            "dx.op.cbufferLoadLegacy.i32",
            fty,
            Some(Attr::ReadOnly),
            false,
        );
        let wave = k.m.function(
            "dx.op.waveReadLaneFirst.i32",
            fty,
            Some(Attr::NoUnwind),
            false,
        );
        let b = k.b;
        let mut g = Gen {
            k,
            rng: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            regs: Vec::new(),
            conds: Vec::new(),
            cur: b,
            input,
            table,
            wave,
            last: None,
        };
        for r in 0..5 {
            let (x, y) = (g.k.m.i32(r), g.k.m.i32(r * 3));
            let v = g.k.m.call(b, input, vec![x, y]);
            g.regs.push(v);
        }
        g.code(3);
        g.code(2);
        // Every register is read at the end.
        for r in 0..g.regs.len() {
            let at = g.k.m.i32(r as u32);
            g.k.m.call(g.cur, g.k.writes, vec![at, g.regs[r]]);
        }
        g.k.m.terminate(g.cur, Term::Ret);
        g.k.m
    }

    #[test]
    fn every_pass_leaves_what_random_programs_store() {
        let mut removed = 0;
        for seed in 1..=400u64 {
            let plain = random_program(seed);
            let inputs: Vec<Vec<u32>> = (0..6u32)
                .map(|n| {
                    (0..7u32)
                        .map(|i| match n {
                            0 => 0,
                            1 => i,
                            2 => u32::MAX - i,
                            _ => {
                                (seed as u32 ^ n)
                                    .wrapping_mul(0x9E37_79B9)
                                    .rotate_left(i * 5)
                                    ^ (i << 28)
                            }
                        })
                        .collect()
                })
                .collect();
            let expected: Vec<_> = inputs.iter().map(|i| run(&plain, i)).collect();
            let written: usize = plain
                .blocks
                .iter()
                .map(|b| b.body.len() + b.phis.len())
                .sum();
            for passes in [
                DEAD,
                FOLD,
                EXT_FOLD,
                REUSE,
                FACTS,
                LOADS,
                FOLD | FACTS,
                FOLD | EXT_FOLD | FACTS,
                FOLD | EXT_FOLD | REUSE | DEAD,
                ALL,
            ] {
                let mut m = random_program(seed);
                m.optimise(passes);
                for (i, want) in inputs.iter().zip(&expected) {
                    assert_eq!(&run(&m, i), want, "seed {seed}, passes {passes}");
                }
                if passes == ALL {
                    let left: usize = m.blocks.iter().map(|b| b.body.len() + b.phis.len()).sum();
                    removed += written - left;
                }
                m.write();
            }
        }
        // The programs are of the wasteful kind: the passes find work.
        assert!(removed > 4000, "{removed} instructions removed");
    }

    #[test]
    fn extended_fold_algebraic_bitwise_shifts_and_strength_reduction() {
        let Kit {
            mut m,
            b,
            reads,
            writes,
            ..
        } = kit();
        let (k0, k1, k2, k3, k4, k5, k6, k8, k16, k20, k32) = (
            m.i32(0),
            m.i32(1),
            m.i32(2),
            m.i32(3),
            m.i32(4),
            m.i32(5),
            m.i32(6),
            m.i32(8),
            m.i32(16),
            m.i32(20),
            m.i32(32),
        );
        let x = m.call(b, reads, vec![k1, k1]);
        let y = m.call(b, reads, vec![k2, k2]);

        // x - x => 0, x ^ x => 0, (x ^ c) ^ c => x
        let sub_self = m.bin(b, Bin::Sub, x, x);
        let xor_self = m.bin(b, Bin::Xor, x, x);
        let xc = m.bin(b, Bin::Xor, x, k5);
        let xcc = m.bin(b, Bin::Xor, xc, k5);
        // (x + c1) + c2 => x + (c1 + c2), (x + c) - c => x, (x - c1) - c2 => x - (c1 + c2)
        let add1 = m.bin(b, Bin::Add, x, k3);
        let add2 = m.bin(b, Bin::Add, add1, k4);
        let add_cancel = m.bin(b, Bin::Sub, add1, k3);
        let sub1 = m.bin(b, Bin::Sub, x, k3);
        let sub2 = m.bin(b, Bin::Sub, sub1, k4);
        let wrap_hi = m.i32(u32::MAX);
        let wrap1 = m.bin(b, Bin::Add, x, wrap_hi);
        let wrap_zero = m.bin(b, Bin::Add, wrap1, k1);
        // (x & c1) & c2 => x & (c1 & c2), (x | c1) | c2 => x | (c1 | c2)
        let m0f = m.i32(0x0F);
        let m3c = m.i32(0x3C);
        let and1 = m.bin(b, Bin::And, x, m0f);
        let and2 = m.bin(b, Bin::And, and1, m3c);
        let or1 = m.bin(b, Bin::Or, x, m0f);
        let or2 = m.bin(b, Bin::Or, or1, m3c);
        // Shifts: (x << 3) << 4 => x << 7, (x >>u 5) >>u 6 => x >>u 11, (x << 20) << 20 => 0
        let shl1 = m.bin(b, Bin::Shl, x, k3);
        let shl2 = m.bin(b, Bin::Shl, shl1, k4);
        let lshr1 = m.bin(b, Bin::LShr, x, k5);
        let lshr2 = m.bin(b, Bin::LShr, lshr1, k6);
        let shl_ov1 = m.bin(b, Bin::Shl, x, k20);
        let shl_ov2 = m.bin(b, Bin::Shl, shl_ov1, k20);
        // Negative control: shift by >= 32 must NOT fold.
        let bad_sh = m.bin(b, Bin::Shl, x, k32);
        let bad_sh2 = m.bin(b, Bin::Shl, bad_sh, k1);
        // Strength reduction: x * 8 => x << 3, x /u 8 => x >>u 3, x %u 16 => x & 15, x %u 1 => 0
        let mul_p2 = m.bin(b, Bin::Mul, x, k8);
        let udiv_p2 = m.bin(b, Bin::UDiv, x, k8);
        let urem_p2 = m.bin(b, Bin::URem, x, k16);
        let urem_1 = m.bin(b, Bin::URem, x, k1);
        // Negative controls: x * 6, x /s 8, x /u 0, x %u 0 must stay as written.
        let mul_6 = m.bin(b, Bin::Mul, y, k6);
        let sdiv_8 = m.bin(b, Bin::SDiv, y, k8);
        let udiv_0 = m.bin(b, Bin::UDiv, y, k0);
        let urem_0 = m.bin(b, Bin::URem, y, k0);

        for (p, q) in [
            (sub_self, xor_self),
            (xcc, add_cancel),
            (add2, sub2),
            (wrap_zero, and2),
            (or2, shl2),
            (lshr2, shl_ov2),
            (bad_sh2, mul_p2),
            (udiv_p2, urem_p2),
            (urem_1, mul_6),
            (sdiv_8, udiv_0),
            (urem_0, k0),
        ] {
            m.call(b, writes, vec![p, q]);
        }
        m.terminate(b, Term::Ret);
        m.optimise(FOLD | EXT_FOLD | DEAD);

        let st = stores(&m, b);
        assert_eq!(st[0], [k0, k0], "x - x and x ^ x fold to 0");
        assert_eq!(st[1], [x, x], "(x ^ c) ^ c and (x + c) - c fold to x");
        assert_eq!(st[3][0], x, "(x + 0xFFFF_FFFF) + 1 wraps to x");
        assert_eq!(st[5][1], k0, "(x << 20) << 20 folds to 0");
        assert_eq!(st[8][0], k0, "x %u 1 folds to 0");

        let k7 = m.i32(7);
        let k11 = m.i32(11);
        let k15 = m.i32(15);
        let k0c = m.i32(0x0C);
        let k3f = m.i32(0x3F);
        let inst_op = |v: V| match v {
            V::Inst(id) => m.insts[id as usize].op.clone(),
            other => panic!("expected inst, got {other:?}"),
        };
        assert!(matches!(inst_op(st[2][0]), Op::Bin(Bin::Add, a, c) if a == x && c == k7));
        assert!(matches!(inst_op(st[2][1]), Op::Bin(Bin::Sub, a, c) if a == x && c == k7));
        assert!(matches!(inst_op(st[3][1]), Op::Bin(Bin::And, a, c) if a == x && c == k0c));
        assert!(matches!(inst_op(st[4][0]), Op::Bin(Bin::Or, a, c) if a == x && c == k3f));
        assert!(matches!(inst_op(st[4][1]), Op::Bin(Bin::Shl, a, c) if a == x && c == k7));
        assert!(matches!(inst_op(st[5][0]), Op::Bin(Bin::LShr, a, c) if a == x && c == k11));
        assert!(matches!(inst_op(st[6][0]), Op::Bin(Bin::Shl, a, c) if a == bad_sh && c == k1));
        assert!(matches!(inst_op(st[6][1]), Op::Bin(Bin::Shl, a, c) if a == x && c == k3));
        assert!(matches!(inst_op(st[7][0]), Op::Bin(Bin::LShr, a, c) if a == x && c == k3));
        assert!(matches!(inst_op(st[7][1]), Op::Bin(Bin::And, a, c) if a == x && c == k15));
        assert!(matches!(inst_op(st[8][1]), Op::Bin(Bin::Mul, a, c) if a == y && c == k6));
        assert!(matches!(inst_op(st[9][0]), Op::Bin(Bin::SDiv, a, c) if a == y && c == k8));
        assert!(matches!(inst_op(st[9][1]), Op::Bin(Bin::UDiv, a, c) if a == y && c == k0));
        assert!(matches!(inst_op(st[10][0]), Op::Bin(Bin::URem, a, c) if a == y && c == k0));
        m.write();
    }

    #[test]
    fn compare_select_zext_fold_and_nan_safety() {
        let Kit {
            mut m,
            b,
            reads,
            writes,
            ..
        } = kit();
        let (i32t, f32t) = (m.int(32), m.float());
        let (k0, k1, k2, k10, k20, k30) = (
            m.i32(0),
            m.i32(1),
            m.i32(2),
            m.i32(10),
            m.i32(20),
            m.i32(30),
        );
        let (t, f) = (m.bool(true), m.bool(false));
        let x = m.call(b, reads, vec![k1, k1]);
        let y = m.call(b, reads, vec![k2, k2]);
        let xf = m.cast(b, Cast::Bitcast, x, f32t);
        let yf = m.cast(b, Cast::Bitcast, y, f32t);

        // xor(icmp(IULT, x, y), true) => icmp(IUGE, x, y)
        let ic = m.cmp(b, pred::IULT, x, y);
        let nic = m.bin(b, Bin::Xor, ic, t);
        // Canonical REUSE: `icmp(IULE, y, x)` equals `icmp(IUGE, x, y)`.
        let mir_ic = m.cmp(b, pred::IULE, y, x);
        // xor(fcmp(OLT, xf, yf), true) => fcmp(UGE, xf, yf)
        let fc = m.cmp(b, pred::OLT, xf, yf);
        let nfc = m.bin(b, Bin::Xor, fc, t);
        // Canonical float REUSE: `fcmp(ULE, yf, xf)` equals `fcmp(UGE, xf, yf)`.
        let mir_fc = m.cmp(b, pred::ULE, yf, xf);
        // Negative control (AMD driver): xor(fcmp(UEQ, xf, yf), true) MUST NOT become fcmp(ONE).
        let ueq = m.cmp(b, pred::UEQ, xf, yf);
        let nueq = m.bin(b, Bin::Xor, ueq, t);
        // Negative control (NaN): fcmp(OEQ, xf, xf) MUST NOT fold to true.
        let nan_eq = m.cmp(b, pred::OEQ, xf, xf);
        let nan_une = m.cmp(b, pred::UNE, xf, xf);

        // select(xor(opaque_bool, true), x, y) => select(opaque_bool, y, x)
        let i1t = m.int(1);
        let opaque_b = m.cast(b, Cast::Trunc, x, i1t);
        let not_opaque = m.bin(b, Bin::Xor, opaque_b, t);
        let sel_flip = m.select(b, not_opaque, x, y);
        // Boolean selects on well-defined values:
        let c2 = m.cmp(b, pred::EQ, x, k10);
        let b2 = m.cmp(b, pred::EQ, y, k20);
        let sel_or = m.select(b, c2, t, b2);
        let sel_and = m.select(b, c2, b2, f);
        let sel_not = m.select(b, c2, f, t);
        // icmp (select(c, k10, k20), k)
        let sel_k = m.select(b, c2, k10, k20);
        let eq_10 = m.cmp(b, pred::EQ, sel_k, k10);
        let eq_20 = m.cmp(b, pred::EQ, sel_k, k20);
        let eq_30 = m.cmp(b, pred::EQ, sel_k, k30);
        // icmp (zext i1 c2, 0/1) & and(zext i1 c2, 1/2)
        let z = m.cast(b, Cast::ZExt, c2, i32t);
        let z_eq0 = m.cmp(b, pred::EQ, z, k0);
        let z_eq1 = m.cmp(b, pred::EQ, z, k1);
        let z_ne1 = m.cmp(b, pred::NE, k1, z);
        let z_and1 = m.bin(b, Bin::And, z, k1);
        let z_and2 = m.bin(b, Bin::And, z, k2);

        let w = |m: &mut Module, v: V| m.cast(b, Cast::ZExt, v, i32t);
        let vals = [
            (w(&mut m, nic), w(&mut m, mir_ic)),
            (w(&mut m, nfc), w(&mut m, mir_fc)),
            (w(&mut m, nueq), w(&mut m, nan_eq)),
            (w(&mut m, nan_une), sel_flip),
            (w(&mut m, sel_or), w(&mut m, sel_and)),
            (w(&mut m, sel_not), w(&mut m, eq_10)),
            (w(&mut m, eq_20), w(&mut m, eq_30)),
            (w(&mut m, z_eq0), w(&mut m, z_eq1)),
            (w(&mut m, z_ne1), z_and1),
            (z_and2, k0),
        ];
        for (p, q) in vals {
            m.call(b, writes, vec![p, q]);
        }
        m.terminate(b, Term::Ret);
        m.optimise(FOLD | EXT_FOLD | REUSE | DEAD);

        let st = stores(&m, b);
        assert_eq!(st[0][0], st[0][1], "icmp IUGE(x,y) merges with IULE(y,x)");
        assert_eq!(st[1][0], st[1][1], "fcmp UGE(xf,yf) merges with ULE(yf,xf)");
        for &id in &m.blocks[b].body {
            if let Op::Cmp(p, _, _) = m.insts[id as usize].op {
                assert_ne!(p, pred::ONE, "must never emit fcmp ONE (AMD driver bug)");
            }
        }
        assert!(matches!(st[2][1], V::Inst(_)), "fcmp OEQ(xf, xf) must stay");
        assert!(matches!(st[3][0], V::Inst(_)), "fcmp UNE(xf, xf) must stay");
        let V::Inst(sf) = st[3][1] else { panic!() };
        assert!(
            matches!(m.insts[sf as usize].op, Op::Select(c, a, z) if (c, a, z) == (opaque_b, y, x))
        );
        let unwrap_zext = |v: V| match v {
            V::Inst(id) => match m.insts[id as usize].op {
                Op::Cast(Cast::ZExt, inner) => inner,
                _ => panic!("expected zext, got {:?}", m.insts[id as usize].op),
            },
            _ => panic!("expected inst, got {v:?}"),
        };
        let V::Inst(or_id) = unwrap_zext(st[4][0]) else { panic!() };
        assert!(matches!(m.insts[or_id as usize].op, Op::Bin(Bin::Or, a, b) if (a, b) == (c2, b2)));
        let V::Inst(and_id) = unwrap_zext(st[4][1]) else { panic!() };
        assert!(matches!(m.insts[and_id as usize].op, Op::Bin(Bin::And, a, b) if (a, b) == (c2, b2)));
        let V::Inst(not_id) = unwrap_zext(st[5][0]) else { panic!() };
        assert!(matches!(m.insts[not_id as usize].op, Op::Cmp(pred::NE, a, k) if (a, k) == (x, k10)));
        assert_eq!(unwrap_zext(st[5][1]), c2, "eq(select(c2,10,20),10) => c2");
        assert_eq!(unwrap_zext(st[6][0]), V::Inst(not_id), "eq(select(c2,10,20),20) => !c2");
        assert_eq!(st[6][1], k0, "eq(select(c2,10,20),30) => false");
        assert_eq!(unwrap_zext(st[7][0]), V::Inst(not_id), "eq(zext c2, 0) => !c2");
        assert_eq!(unwrap_zext(st[7][1]), c2, "eq(zext c2, 1) => c2");
        assert_eq!(unwrap_zext(st[8][0]), V::Inst(not_id), "ne(1, zext c2) => !c2");
        assert_eq!(st[8][1], z, "and(zext c2, 1) => zext c2");
        assert_eq!(st[9][0], k0, "and(zext c2, 2) => 0");
        m.write();
    }

    #[test]
    fn default_63_leaves_ext_fold_off() {
        let build = || {
            let Kit {
                mut m,
                b,
                reads,
                writes,
                ..
            } = kit();
            let i32t = m.int(32);
            let (k1, k2, k8, k10) = (m.i32(1), m.i32(2), m.i32(8), m.i32(10));
            let t = m.bool(true);
            let x = m.call(b, reads, vec![k1, k1]);
            let y = m.call(b, reads, vec![k2, k2]);
            let mul8 = m.bin(b, Bin::Mul, x, k8);
            let sub_self = m.bin(b, Bin::Sub, x, x);
            let c1 = m.cmp(b, pred::EQ, x, k10);
            let c2 = m.cmp(b, pred::EQ, y, k10);
            let sel = m.select(b, c1, t, c2);
            let sel_i32 = m.cast(b, Cast::ZExt, sel, i32t);
            m.call(b, writes, vec![mul8, sub_self]);
            m.call(b, writes, vec![sel_i32, k1]);
            m.terminate(b, Term::Ret);
            m
        };
        assert_eq!(DEFAULT, 63);
        let mut base = build();
        base.optimise(DEFAULT);
        let st_base = stores(&base, 0);
        let op_of = |m: &Module, v: V| match v {
            V::Inst(id) => m.insts[id as usize].op.clone(),
            other => panic!("expected inst, got {other:?}"),
        };
        assert!(matches!(op_of(&base, st_base[0][0]), Op::Bin(Bin::Mul, ..)));
        assert!(matches!(op_of(&base, st_base[0][1]), Op::Bin(Bin::Sub, ..)));
        let Op::Cast(Cast::ZExt, V::Inst(sel_id)) = op_of(&base, st_base[1][0]) else {
            panic!("expected zext");
        };
        assert!(matches!(base.insts[sel_id as usize].op, Op::Select(..)));

        let mut ext = build();
        ext.optimise(DEFAULT | EXT_FOLD);
        let st_ext = stores(&ext, 0);
        assert!(matches!(op_of(&ext, st_ext[0][0]), Op::Bin(Bin::Shl, ..)));
        assert_eq!(st_ext[0][1], ext.i32(0));
        let Op::Cast(Cast::ZExt, V::Inst(or_id)) = op_of(&ext, st_ext[1][0]) else {
            panic!("expected zext");
        };
        assert!(matches!(ext.insts[or_id as usize].op, Op::Bin(Bin::Or, ..)));
    }

    #[test]
    fn fcmp_opposite_matches_ieee754_for_all_predicates_including_nans() {
        let samples = [
            0.0f32.to_bits(),
            (-0.0f32).to_bits(),
            1.0f32.to_bits(),
            (-2.5f32).to_bits(),
            f32::INFINITY.to_bits(),
            f32::NEG_INFINITY.to_bits(),
            f32::NAN.to_bits(),
            0x7F80_0001u32, // sNaN
        ];
        for p in 0..=15u8 {
            let build = || {
                let mut k = kit();
                let (i32t, f32t) = (k.m.int(32), k.m.float());
                let fty = k.m.func_ty(i32t, vec![i32t, i32t]);
                let inp = k.m.function("dx.op.loadInput.i32", fty, Some(Attr::ReadNone), false);
                let (k0, k1) = (k.m.i32(0), k.m.i32(1));
                let a = k.m.call(k.b, inp, vec![k0, k0]);
                let b = k.m.call(k.b, inp, vec![k1, k0]);
                let af = k.m.cast(k.b, Cast::Bitcast, a, f32t);
                let bf = k.m.cast(k.b, Cast::Bitcast, b, f32t);
                let c = k.m.cmp(k.b, p, af, bf);
                let yes = k.m.bool(true);
                let nc = k.m.bin(k.b, Bin::Xor, c, yes);
                let z = k.m.cast(k.b, Cast::ZExt, nc, i32t);
                k.m.call(k.b, k.writes, vec![k0, z]);
                k.m.terminate(k.b, Term::Ret);
                k.m
            };
            let plain = build();
            let mut opt = build();
            opt.optimise(FOLD | EXT_FOLD | REUSE | DEAD);
            for &x in &samples {
                for &y in &samples {
                    assert_eq!(
                        run(&opt, &[x, y]),
                        run(&plain, &[x, y]),
                        "pred {p}, x={x:#x}, y={y:#x}"
                    );
                }
            }
        }
    }

    #[test]
    fn canonical_facts_propagate_flipped_comparisons() {
        let Diamond {
            k: Kit { mut m, writes, .. },
            x,
            y,
            then,
            other,
            ..
        } = diamond();
        // Entry branches on `x <u y`.
        // In `then`: `y >u x` is true, `y <=u x` is false, `x >=u y` is false.
        let flipped_true = m.cmp(then, pred::IUGT, y, x);
        let flipped_false = m.cmp(then, pred::IULE, y, x);
        let s1 = m.select(then, flipped_true, x, y);
        let s2 = m.select(then, flipped_false, x, y);
        m.call(then, writes, vec![s1, s2]);
        // In `other`: `y <=u x` is true, `y >u x` is false.
        let o1 = m.cmp(other, pred::IULE, y, x);
        let o2 = m.cmp(other, pred::IUGT, y, x);
        let s3 = m.select(other, o1, x, y);
        let s4 = m.select(other, o2, x, y);
        m.call(other, writes, vec![s3, s4]);
        m.optimise(FOLD | EXT_FOLD | FACTS | DEAD);
        assert_eq!(stores(&m, then), [[x, y]]);
        assert_eq!(stores(&m, other), [[x, y]]);
        m.write();
    }
}
