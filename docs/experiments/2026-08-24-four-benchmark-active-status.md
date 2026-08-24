# Four-benchmark active method status

## Scope

SPIn is outside the short-term critical path because only 9/10 scenes are
available.  No SPIn process is running and its existing assets are preserved.
The active contracts are LERF2D, LERF3D, ScanNet OVS and NVOS.

## Current validated development rows

| benchmark | retained result | change over the relevant retained baseline | status |
|---|---:|---:|---|
| LERF2D full4 | sample-micro mIoU `0.39584`, LocAcc `0.87981` | `+0.08167` mIoU, unchanged LocAcc | identity--extent posterior retained; 4/4 scenes positive |
| LERF3D full4 | mIoU `0.43730`, Acc@.25 `0.67788`, Acc@.50 `0.48077` | `+0.04047` mIoU over the prior typed posterior | latent proposal/null readout retained for 3D only |
| ScanNet OVS paper8 compact student | 19/15/10 mIoU `0.36027/0.35957/0.46626` | `+0.00414/+0.00658/+0.00672` over restored L512 baseline | compact aggregate gain; scene0400 regresses |
| ScanNet OVS paper8 exact native region | 19/15/10 mIoU `0.36630/0.36301/0.46866` | `+0.01017/+0.01001/+0.00912` | 24/24 scene-split mIoU positive; native sidecar not fully compressed |
| NVOS RGB-assisted full8 | macro IoU `0.92555`, pixel accuracy `0.98553` | `+0.10793` over Method-v1 target-only | no LUDVIG; 5 video branches, 3 safe fallbacks, no scene regression |

All rows are development evidence.  None is an SOTA or outcome-blind final-test
claim.

## Unified conclusion

The evidence supports a shared persistent RADIO-anchored latent plus typed
capability compilers.  It rejects the stronger claim that post-MPR RADIO
adaptors alone are sufficient for every task variable.

- ScanNet improves when category capability and object-region identity/extent
  are computed natively in 2D before exact marginal lifting.
- NVOS improves when identity, complete prompt extent, temporal transport and
  reliability rejection are explicit typed variables.
- LERF improves from the same identity--extent factorization, but its current
  source-SAM proposal graph is not a stable complete physical-instance
  authority.  Native DINO association, learned membership, denser proposals
  and automatic video tracks have all failed disjoint-source gates.

The main unresolved method problem is therefore not generic RADIO feature
reconstruction.  It is compactly representing and transporting complete
object membership while preserving text identity margins.  For ScanNet this
appears as a native-teacher-to-L512 query-margin compression gap; for LERF it
appears earlier, as a missing independent 3D physical-instance teacher.

## External proposal closure

The latest external multi-teacher proposal is partially, not fully, closed.
Native DINO/SAM/SigLIP extraction, correct-level exact MPR, DINO A/B/C,
source-heldout LERF membership, ScanNet native region supervision and NVOS
prompt-region memory are implemented.  A single joint optimization of the same
L512 row under all three native teachers is not implemented.  Current matched
DINO evidence selects a frozen RADIO latent, so an undifferentiated joint loss
is not justified; the next joint experiment must optimize query-variable
responses and object membership, not simply add dense cosine losses.
