# Single-region candidate availability and admission diagnostic

Fixed eight NVOS development requests, seven frozen fields, old 120 source views.
No training, new source state, threshold fitting, default changes or model promotion.
Enumerate every existing positive relation coordinate as a Gaussian subset B[:,r]>0
intersected with member_available. Re-render each subset with its own transmittance
at the existing 192x256 projection resolution, saving float32 alpha before labels.
Save existing scale compatibility, positive-support admission coefficients and view
IDs. A full eight-case generation/receipt barrier precedes all label access.

After that barrier, nearest-resize target masks to 192x256. At fixed alpha>0.5,
report best single-region IoU in all / compatible / admitted pools and the existing
maximum-coefficient candidate (ties lowest index). Oracle winners are diagnostic
only. Re-render the old local and three union coverage controls at exactly this
resolution. These coarse diagnostic scores are NOT official NVOS results; do not
compare directly to previously reported benchmark IoUs or v10. A poor single-region
oracle cannot rule out multiple-region object composition or a different field.
A high union recall cannot establish any single good object hypothesis.

Development labels were seen in earlier studies; this is not a blind final test.
All source code and inputs are hashed. No source RGB, masks or tracks are needed
by generation. This runner does not claim an OS-level runtime access audit.
