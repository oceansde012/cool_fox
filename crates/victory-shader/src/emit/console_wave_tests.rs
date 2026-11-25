//! Console wave64 compute programs checked lane by lane against what the console gives,
//! through the whole translation ([`super::emit`], with its fallbacks) and on the host GPU:
//! scalars a divergent loop or branch leaves, masks counted after branches, groups of
//! several console waves, whole-wave reductions, `ds_bpermute_b32` / `ds_permute_b32`
//! (RDNA ISA 10.4.4: each 32-lane half of a wave64 on its own). The programs and the
//! expected values are those of Belobug/VIctory-Belobug-AMD PRs #68 and #128.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::dxbc::Container;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// An empty DXBC container: a compute program needs none of its parts.
fn empty_container() -> Vec<u8> {
    let mut c = b"DXBC".to_vec();
    c.extend([0u8; 16]);
    c.extend(1u16.to_le_bytes());
    c.extend(0u16.to_le_bytes());
    c.extend(32u32.to_le_bytes());
    c.extend(0u32.to_le_bytes());
    c
}

/// `threads` x 1 x 1 console wave64 threads per group, a UAV table in user SGPR 0 and the
/// group id in s1: lane t of group g stores what `body` leaves in v2 (v0 = t) at dword
/// g * threads + t.
fn program(body: &[u32], threads: u32) -> NativeProgram {
    let mut words = vec![
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF408_0101,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[2:3], null
        0x9308_FF01,
        threads * 4, // s_mul_i32 s8, s1, threads * 4
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];
    words.extend_from_slice(body);
    words.extend_from_slice(&[
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0x4A06_0608, // v_add_nc_u32 v3, s8, v3
        0xE070_1000,
        0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
        S_ENDPGM,
    ]);
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 16,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1 | 1 << 7, // one user SGPR, then the group id x
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// The whole translation for a host whose waves have at most `host_wave` lanes.
fn translate(native: &NativeProgram, host_wave: u32) -> Result<HostShader, ShaderError> {
    let bytes = empty_container();
    let c = Container::parse(&bytes)?;
    let p = crate::spill::build(native)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, native)?;
    let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
    let r = super::emit(&c, native, &p, &st, &io, &res);
    lower::TEST_HOST_WAVE.set(was);
    r
}

/// What `native` stores in 2 groups on the host GPU, or why it did not run.
fn run(native: &NativeProgram, host: &Host, host_wave: u32) -> Result<Vec<u32>, String> {
    let shader = translate(native, host_wave).map_err(|e| format!("refused: {e}"))?;
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .map_err(|e| format!("invalid DXIL: {e}"))?;
    }
    let (root, pso) = host
        .compute_pipeline(&shader)
        .map_err(|e| format!("pipeline: {e}"))?;
    let StageIo::Cs(io) = &native.io else {
        unreachable!()
    };
    let n = io.threads[0] * 2;
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(n)],
                samplers: &[],
                groups: [2, 1, 1],
            },
        )
        .map_err(|e| format!("dispatch: {e}"))?;
    Ok(out.into_iter().next().unwrap_or_default())
}

/// Threads of console wave `w` in a group of `n`.
fn wave_threads(n: u32, w: u32) -> u32 {
    n.saturating_sub(64 * w).min(64)
}

/// v1 = the lane's place in its console wave (t & 63), v5 = its console wave (t >> 6).
const LANE_AND_WAVE: [u32; 2] = [
    0x3602_00BF, // v_and_b32 v1, 63, v0
    0x2C0A_0086, // v_lshrrev_b32 v5, 6, v0
];

/// The cases: (name, body, group sizes, lane t's value in a group of n).
type Want = fn(u32, u32) -> u32;
fn cases() -> Vec<(&'static str, Vec<u32>, Vec<u32>, Want)> {
    let mut v: Vec<(&'static str, Vec<u32>, Vec<u32>, Want)> = Vec::new();

    // A scalar counted in a divergent loop (lane l of its wave loops l + 1 times), read after
    // it: the console runs the loop until its last lane leaves, so every lane reads 64.
    v.push((
        "scalar counted in a divergent loop, read after it",
        vec![
            0x3602_00BF, // v_and_b32 v1, 63, v0
            0x4A02_0281, // v_add_nc_u32 v1, 1, v1   (lane l loops l + 1 times)
            0xBE82_0480, // s_mov_b64 s[2:3], 0
            0xBE89_0380, // s_mov_b32 s9, 0
            0x8109_8109, // loop: s_add_i32 s9, s9, 1
            0x4E02_0281, // v_subrev_nc_u32 v1, 1, v1
            0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
            0x8882_026A, // s_or_b64 s[2:3], vcc, s[2:3]
            0x8AFE_027E, // s_andn2_b64 exec, exec, s[2:3]
            0xBF89_FFFA, // s_cbranch_execnz loop
            0x88FE_027E, // s_or_b64 exec, exec, s[2:3]
            0x7E04_0209, // v_mov_b32 v2, s9
        ],
        vec![64, 128],
        |t, n| wave_threads(n, t / 64),
    ));

    // A divergent loop's exit mask (lanes that left in earlier iterations are not running
    // where it is last written), counted after the loop: every lane of the console wave.
    v.push((
        "loop exit mask counted after the loop",
        vec![
            0x3602_00BF, // v_and_b32 v1, 63, v0
            0x4A02_0281, // v_add_nc_u32 v1, 1, v1
            0xBE82_0480, // s_mov_b64 s[2:3], 0
            0x4E02_0281, // loop: v_subrev_nc_u32 v1, 1, v1
            0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
            0x8882_026A, // s_or_b64 s[2:3], vcc, s[2:3]
            0x8AFE_027E, // s_andn2_b64 exec, exec, s[2:3]
            0xBF89_FFFB, // s_cbranch_execnz loop
            0x88FE_027E, // s_or_b64 exec, exec, s[2:3]
            0xBE89_1002, // s_bcnt1_i32_b64 s9, s[2:3]
            0x7E04_0209, // v_mov_b32 v2, s9
        ],
        vec![64, 128],
        |t, n| wave_threads(n, t / 64),
    ));

    // An if/else on the lane, s9 set on both sides: the console wave runs each side some lane
    // takes, the else side last, so every lane reads 2 where some lane took it, else 1.
    // Wave w's lanes below 16 (w + 1) take the then side.
    v.push((
        "scalar set on both sides of a divergent if/else",
        {
            let mut b = LANE_AND_WAVE.to_vec();
            b.extend([
                0x4A0C_0A81, // v_add_nc_u32 v6, 1, v5
                0x340C_0C84, // v_lshlrev_b32 v6, 4, v6
                0xBE89_0380, // s_mov_b32 s9, 0
                0x7D88_0306, // v_cmp_gt_u32 vcc, v6, v1
                0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
                0xBF88_0001, // s_cbranch_execz else
                0xBE89_0381, // s_mov_b32 s9, 1
                0x89FE_0A7E, // else: s_xor_b64 exec, exec, s[10:11]
                0xBF88_0001, // s_cbranch_execz join
                0xBE89_0382, // s_mov_b32 s9, 2
                0x88FE_0A7E, // join: s_or_b64 exec, exec, s[10:11]
                0x7E04_0209, // v_mov_b32 v2, s9
            ]);
            b
        },
        vec![64, 128, 256, 96],
        |t, n| {
            let w = t / 64;
            if wave_threads(n, w) > 16 * (w + 1) {
                2
            } else {
                1
            }
        },
    ));

    // A mask with lanes from before a divergent branch (0-15) and lanes added inside it
    // (32-47), counted after it: 32.
    v.push((
        "mask grown inside a branch, counted after it",
        vec![
            0x3602_00BF, // v_and_b32 v1, 63, v0
            0x7D88_0290, // v_cmp_gt_u32 vcc, 16, v1      (lanes 0-15)
            0xBE8C_046A, // s_mov_b64 s[12:13], vcc
            0x7D86_02A0, // v_cmp_le_u32 vcc, 32, v1      (lanes 32-63)
            0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
            0xBF88_0002, // s_cbranch_execz join
            0x7D88_02B0, // v_cmp_gt_u32 vcc, 48, v1      (of those: 32-47)
            0x888C_0C6A, // s_or_b64 s[12:13], vcc, s[12:13]
            0x88FE_0A7E, // join: s_or_b64 exec, exec, s[10:11]
            0xBE89_100C, // s_bcnt1_i32_b64 s9, s[12:13]
            0x7E04_0209, // v_mov_b32 v2, s9
        ],
        vec![64, 128],
        |_, _| 32,
    ));

    // Masks counted per console wave: s_bcnt1 of the lanes below 40 - w.
    v.push((
        "s_bcnt1 of lanes below 40 - wave",
        {
            let mut b = LANE_AND_WAVE.to_vec();
            b.extend([
                0x4C0C_0AA8, // v_sub_nc_u32 v6, 40, v5
                0x7D88_0306, // v_cmp_gt_u32 vcc, v6, v1
                0xBE89_106A, // s_bcnt1_i32_b64 s9, vcc
                0x7E04_0209, // v_mov_b32 v2, s9
            ]);
            b
        },
        vec![64, 128, 256, 96, 80, 112, 48, 16],
        |t, n| (40 - t / 64).min(wave_threads(n, t / 64)),
    ));

    // Lane 40 of the console wave, in its upper half.
    v.push((
        "v_readlane 40",
        vec![
            0xD760_0009,
            0x0001_5100, // v_readlane_b32 s9, v0, 40
            0x7E04_0209, // v_mov_b32 v2, s9
        ],
        vec![64, 128, 256, 112],
        |t, _| (t & !63) + 40,
    ));

    // WaveActiveAnyTrue choosing a value: only lane 36 (upper half) sets the mask.
    v.push((
        "SCC of (t == 36) & exec chooses 5",
        vec![
            0x7D84_00FF,
            36,          // v_cmp_eq_u32 vcc, 36, v0
            0x878C_7E6A, // s_and_b64 s[12:13], vcc, exec
            0x8509_8785, // s_cselect_b32 s9, 5, 7
            0x7E04_0209, // v_mov_b32 v2, s9
        ],
        vec![64],
        |_, _| 5,
    ));

    // ds_bpermute_b32 naming lane t ^ 33 (the other half) reads t ^ 1 (RDNA ISA 10.4.4).
    v.push((
        "ds_bpermute of t ^ 33 reads t ^ 1",
        vec![
            0x3A08_00A1, // v_xor_b32 v4, 33, v0
            0x3408_0882, // v_lshlrev_b32 v4, 2, v4
            0xDACC_0000,
            0x0200_0004, // ds_bpermute_b32 v2, v4, v0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        ],
        vec![64, 128, 96],
        |t, _| t ^ 1,
    ));

    // ds_permute_b32: lanes 2d and 2d + 1 of each half send to lane d of their half; the
    // higher one's value stays, lanes 16-31 of each half get 0.
    v.push((
        "ds_permute to (t & 31) >> 1",
        vec![
            0x3608_009F, // v_and_b32 v4, 31, v0
            0x2C08_0881, // v_lshrrev_b32 v4, 1, v4
            0x3808_08A0, // v_or_b32 v4, 32, v4
            0x3408_0882, // v_lshlrev_b32 v4, 2, v4
            0xDAC8_0000,
            0x0200_0004, // ds_permute_b32 v2, v4, v0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        ],
        vec![64, 128, 96],
        |t, n| {
            let (base, d) = (t & !31, t & 31);
            (0..32)
                .map(|k| base + k)
                .filter(|&s| s < n && (s & 31) >> 1 == d)
                .max()
                .unwrap_or(0)
        },
    ));

    // The same inside a branch only even lanes take: the odd lanes they name do not run the
    // instruction and give 0; the odd lanes keep their 7.
    let even = |n: u32| {
        vec![
            0x7E04_0287,     // v_mov_b32 v2, 7
            0x360A_0081,     // v_and_b32 v5, 1, v0
            0x7D84_0A80,     // v_cmp_eq_u32 vcc, 0, v5
            0xBE8A_246A,     // s_and_saveexec_b64 s[10:11], vcc
            0xBF88_0000 | n, // s_cbranch_execz +n
        ]
    };
    v.push((
        "even lanes: ds_bpermute of t ^ 33",
        {
            let mut b = even(5);
            b.extend([
                0x3A08_00A1,
                0x3408_0882,
                0xDACC_0000,
                0x0200_0004,
                0xBF8C_C07F,
            ]);
            b.push(0x88FE_0A7E); // s_or_b64 exec, exec, s[10:11]
            b
        },
        vec![64, 128],
        |t, _| if t & 1 == 0 { 0 } else { 7 },
    ));
    v.push((
        "even lanes: ds_permute to (t & 31) >> 1",
        {
            let mut b = even(7);
            b.extend([
                0x3608_009F,
                0x2C08_0881,
                0x3808_08A0,
                0x3408_0882,
                0xDAC8_0000,
                0x0200_0004,
                0xBF8C_C07F,
            ]);
            b.push(0x88FE_0A7E);
            b
        },
        vec![64, 128],
        |t, n| {
            if t & 1 == 1 {
                return 7;
            }
            let (base, d) = (t & !31, t & 31);
            (0..32)
                .map(|k| base + k)
                .filter(|&s| s < n && s & 1 == 0 && (s & 31) >> 1 == d)
                .max()
                .unwrap_or(0)
        },
    ));
    v
}

/// Cases that do not give the console's values yet (each half-wave's own), run by
/// [`console_wave64_programs_not_yet_exact`]: a loop's exit mask grows by ballots inside the
/// loop, which two host waves cannot exchange where lanes leave the loop at different times
/// (no group barrier there). The translation's last resort, each half on its own, takes it.
const OPEN: &[&str] = &["loop exit mask counted after the loop"];

/// Every case on this machine's GPU, translated for its own wave size: a line per case and
/// group size with what came out, and a failure if any is not the console's.
#[test]
fn console_wave64_programs_give_the_consoles_values() {
    check(false);
}

/// The [`OPEN`] cases, which are expected to fail until they are done.
#[test]
#[ignore = "not exact yet: see OPEN"]
fn console_wave64_programs_not_yet_exact() {
    check(true);
}

/// Every case translates and validates for a host of 64-lane waves (an AMD card), and asks
/// for them where it reads the console wave's scalars or masks. (Not run here: a 32-lane
/// card takes no `WaveSize(64)`.)
#[test]
fn console_wave64_programs_translate_for_64_lane_hosts() {
    for (name, body, sizes, _) in cases() {
        for n in sizes {
            let native = program(&body, n);
            let shader =
                translate(&native, 64).unwrap_or_else(|e| panic!("{name}, {n} threads: {e}"));
            if let Ok(mut v) =
                validator::Validator::load(validator::Validator::default_path().as_deref())
            {
                v.validate(&mut shader.dxil.clone())
                    .unwrap_or_else(|e| panic!("{name}, {n} threads: {e}"));
            }
        }
    }
}

fn check(open: bool) {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return;
    };
    let host_wave = std::env::var("VICTORY_TEST_HOST_WAVE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let mut wrong = Vec::new();
    for (name, body, sizes, want) in cases() {
        if OPEN.contains(&name) != open {
            continue;
        }
        for n in sizes {
            let native = program(&body, n);
            let expect: Vec<u32> = (0..2).flat_map(|_| (0..n).map(|t| want(t, n))).collect();
            let line = match run(&native, &host, host_wave) {
                Ok(out) if out == expect => format!("exact     {name}, {n} threads"),
                Ok(out) => {
                    let bad = out.iter().zip(&expect).filter(|(a, b)| a != b).count();
                    let first = out
                        .iter()
                        .zip(&expect)
                        .position(|(a, b)| a != b)
                        .unwrap_or(0);
                    format!(
                        "WRONG     {name}, {n} threads: {bad} of {} lanes (lane {first}: {} for {})",
                        expect.len(),
                        out[first],
                        expect[first]
                    )
                }
                Err(e) => format!("NOT RUN   {name}, {n} threads: {e}"),
            };
            eprintln!("{line}");
            if !line.starts_with("exact") {
                wrong.push(line);
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} cases differ from the console:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
