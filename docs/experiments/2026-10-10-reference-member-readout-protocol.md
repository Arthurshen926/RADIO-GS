# Reference evidence is admission evidence, not whole-object probability

Follow-up to completed reference_member_evidence_v1, frozen001e828: actual
selection improves84.3701->86.9550%, but continuous response82.9221->82.5562%.
This is sequential development on the same eight cases, not independent testing.

Same seven fields/120 sources, same retained positive-SAM query masks and inherited
RGB/SAM generation cost. Both arms run the exact same greedy composition and
within-region background-majority exclusion. No new identity function, Gaussian
selection, input condition, training or state; no per-case or threshold search.

- fraction reproduces conditioned_members: selected Gaussian scores are their
  reference foreground fractions (unsupported rows retain their prior), including
  nonzero scores below0.5 for rejected rows. These are not calibrated probabilities.
- Primary admission uses the resulting binary membership as BOTH 2D readout input
  and actual-set membership. The reference fraction is used only to accept/reject
  composed members; it no longer masquerades as whole-object confidence. Exact
  ties and zero-evidence members remain selected as in the previous protocol.

All16 predictions before any new scoring; eight fraction baseline selected arrays
must match001e828 exactly, other payloads within1e-6. All eight admission and
fraction actual selected arrays AND actual rendered responses must match exactly.
Thus any response change is ONLY a readout effect, not further3D scope recovery.
Report both formal IoUs, precision, recall, AP and all cases. Binary membership
is still not a calibrated posterior and may lose ranking quality; no default
promotion or five-task recovery claim. Baseline source protocol and hash records
remain immutable; frozen original code is available at001e828.
