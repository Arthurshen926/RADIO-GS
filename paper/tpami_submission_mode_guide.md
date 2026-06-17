# TPAMI Submission Mode Guide

The current manuscript package is prepared as an **anonymous review draft**:

- Main manuscript: `paper/radio_gs_tpami.pdf`
- Supplement: `paper/radio_gs_tpami_supplement.pdf`
- Author field in both PDFs: `Anonymous Authors`
- No acknowledgements, funding details, or identifying author metadata are
  included in the review PDFs.

This is the safest default because IEEE Computer Society guidance requires
authors to ensure that materials submitted for double-anonymous review do not
reveal author identities. If the TPAMI ScholarOne route used for this submission
instead requests a single-blind manuscript, fill the non-anonymous metadata
below and regenerate both PDFs before upload.

## Non-Anonymous Metadata To Fill If Required

Use this block to replace the anonymous author line in
`paper/radio_gs_tpami.tex` and `paper/radio_gs_tpami_supplement.tex` when the
submission portal explicitly requests author-visible manuscripts.

```latex
\author{First Author, Second Author, and Third Author%
\thanks{First Author is with Department, University, City, Country.
E-mail: first.author@example.edu.}%
\thanks{Second Author is with Department, University, City, Country.
E-mail: second.author@example.edu.}%
\thanks{Third Author is with Department, University, City, Country.
E-mail: third.author@example.edu.}%
\thanks{Corresponding author: First Author.}}
```

Also add an acknowledgement section before the bibliography if the portal
requires it in the submitted PDF:

```latex
\section*{Acknowledgments}
This work was supported by <funding source>. The authors thank <names> for
<specific contribution>. The authors used <AI/tool disclosure if required by
IEEE policy>.
```

## Upload Decision Rule

1. If the portal says **double-anonymous**, upload the current anonymous PDFs and
   keep author names, funding, acknowledgements, and conflicts only in the
   portal metadata fields.
2. If the portal says **single-blind** or asks for author-visible manuscripts,
   fill the metadata above, rebuild the PDFs, rerun the verification checklist
   in `paper/tpami_submission_checklist.md`, and upload the non-anonymous PDFs.
3. If the portal accepts source files at initial submission, inspect source
   metadata and comments before upload. The anonymous PDFs are clean, but source
   packages may contain filesystem paths, comments, or repository names that
   reveal authorship or environment details. Do not include `paper/archive/` in
   the uploaded source package; it contains legacy working drafts retained only
   for project traceability.

## Review-PDF Anonymity Checks

Run these before a double-anonymous upload:

```bash
cd /root/RADIO-GS
rg -n "Anonymous Authors|Acknowledgments|E-mail|University|Institute|Grant|ORCID" \
  paper/radio_gs_tpami.tex paper/radio_gs_tpami_supplement.tex
```

Expected state for the current anonymous review package:

- `Anonymous Authors` appears in the author blocks.
- No real author names, emails, affiliations, ORCIDs, funding agencies, or
  acknowledgements appear in the review PDFs.

For a non-anonymous upload, reverse the expectation: author and affiliation
metadata should be present and the manuscript should no longer say
`Anonymous Authors`.
