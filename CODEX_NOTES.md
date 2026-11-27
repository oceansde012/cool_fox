# Vice-Tay A/B integration

The integration starts at Vice-Tay `de7c29cbf9ae4414b166be6256fc8b7e67378129`. All four requested branches were merged in order, each with two parents. The feature switches are off when unset.

## Merge commits

| Branch | Incoming tip | Result |
| --- | --- | --- |
| `origin/vlad/vram-peak` | `8f618e6f` | Merged as `44a8a0a2572a2700747d4cf10e53d84fe81ae03c` |
| `origin/vlad/perf-rdpool` | `d3b6dbd4` | Merged as `c6e684606024204242b6a1c5b040f37fdeb11b40` |
| `origin/vlad/perf-lookup` | `a961871b` with `23cfe56e` | Merged as `b4027db3f8bed562049a68829be15583b7df99cc` |
| `origin/vlad/perf-pr3353` | `7bb03fec` with `76e5b5c6` | Merged as `bfea7bc4982443debed40158ef944cbcef885fa9` |

## Conflict resolutions

- **VRAM residency and reservation**. In `host.rs`, kept Vice-Tay’s adapter logging and added the opt-in reservation after it. In `resource.rs`, kept the early return for pooled placed textures, reserved and committed creation, and error propagation. The priority hook runs for successful committed textures after residency and mapping. The UAV buffer hook applies to successful committed buffers. The two controls are `VICTORY_RESIDENCY_PRIORITY=1` and `VICTORY_VRAM_RESERVE_MB=max|N`.
- **Readback pool**. Kept Vice-Tay’s `texback.rs`, including fresh tickets, reused scratch storage, and use of the existing host-side pool. That pool already has exclusive leases through recording retirement, fence protections, and allocation statistics. It keeps Vice-Tay’s explicit `VICTORY_READBACK_POOL=1` enable check and `VICTORY_READBACK_POOL_MB` cap, which defaults to 256 MiB. Ported the incoming size-class test to the existing allocator in `host.rs`.
- **Lookup cache**. In `shadow.rs`, kept Vice-Tay’s window budgeting, oldest-first victim selection, existing quick lookup, and renderer-specific window placement. Added the `VICTORY_LOOKUP_CACHE=1` weak-reference cache and generation checks for registered buffers and loose windows. The window sweep invalidates cache hits and rechecks whether candidate windows are idle before choosing victims.
- **First-use pipeline wait**. Kept the existing pipeline-skip entry and added the incoming `VICTORY_PIPELINE_FIRST_WAIT=1` entry in `docs/README-fxceless.md`. The pipeline implementation and both wait tests merged automatically. The opt-in wait can spend up to 10 seconds on a pipeline’s first use.

The runtime changes cover allocation policy, lookup caching, and the optional pipeline wait. Residency priority has two call sites, both for committed resources. Source changes came from the requested refs and Vice-Tay. Validation covered builds and automated tests. The game executable `victory.exe` was never launched.

## Validation

Cargo ran offline. All four `cargo build --locked --release -p victory` commands exited successfully. The per-merge `cargo test --locked --release -p victory-gpu --no-fail-fast` runs used the default switch settings.

| Merge | Passed | Failed | Ignored | Follow-up |
| --- | ---: | ---: | ---: | --- |
| VRAM | 461 | 1 | 42 | `watch::tests::what_is_put_in_beside_the_guards_is_told_at_every_view` failed at `published(two, two + BLOCK)`. An isolated rerun passed. |
| Readback | 463 | 0 | 42 | |
| Lookup | 465 | 0 | 42 | With `VICTORY_LOOKUP_CACHE=1`, both focused tests passed and re-registration ran on a D3D12 host device. |
| Pipeline wait | 467 | 0 | 42 | |

The single workspace run used `cargo test --locked --release --workspace --no-fail-fast` with the test filters below. Cargo exited 0. Totals were **1198 passed, 0 failed, 54 ignored, and 9 filtered out**. There were no failing workspace tests.

The workspace invocation used libtest `--skip` filters for these tests.

- `census_accepts_a_valid_path_with_no_pe_images` launches `victory.exe`.
- `census_fails_when_a_requested_path_is_missing` launches `victory.exe`.
- `warm_used_with_only_a_shader_cache_reports_no_recorded_pipelines` launches `victory.exe`.
- `omitted_and_positive_worker_counts_reach_dispatch` launches `victory.exe`.
- `zero_workers_are_rejected_before_dispatch` launches `victory.exe`.
- `host::imp::tests::loading_a_host_dll_does_not_block_thread_attach` intentionally ends a child process or can terminate it on timeout.
- `modules::xgamesave::tests::a_kill_at_any_step_leaves_old_or_new` intentionally ends a child process or can terminate it on timeout.
- `modules::lifecycle::tests::direct_close_suspends_and_exits` intentionally ends a child process or can terminate it on timeout.
- `thunks::tests::a_trapped_import_is_in_the_crash_log` intentionally ends a child process or can terminate it on timeout.

The existing warm-worker unit tests were still included. Their cleanup calls `Child::kill()` on dummy `cmd.exe` children during error-path tests and discards the call result. No standalone process-kill command was issued.

Raw logs are in `B:/cool_fox/cx-vtall/target/vt-all-evidence`.

## Local checkout and transfer

Sandbox permissions denied Git writes to the original worktree’s metadata under `B:/cool_fox/src/.git/worktrees/cx-vtall`. The completed `codex/vt-all` branch and release build are in `B:/cool_fox/cx-vtall/target/vt-all-checkout`. The original worktree’s branch remains at `de7c29cb`. Its root copy of these notes is untracked there and is committed in the writable checkout.

An offline fetch refreshed the cached `origin/main` snapshot from the primary checkout’s refs. The primary `main` and cached `origin/main` both resolve to `f4b48ad766ddbec1a1867022cd42f7645b0dd55e` with zero commits of divergence. The primary checkout’s existing untracked build directories were preserved. No changes were pushed.

The bundle `target/vt-all-integration.bundle` contains the integration branch and uses the five already-fetched input commits as prerequisites. To bring it into the original worktree with Git write access, move the untracked root copy of these notes aside, fetch the bundle, and fast-forward `codex/vt-all`.

```powershell
Move-Item -LiteralPath CODEX_NOTES.md -Destination target/CODEX_NOTES.pre-import.md
git fetch --no-tags ./target/vt-all-integration.bundle refs/heads/codex/vt-all
git merge --ff-only FETCH_HEAD
```
