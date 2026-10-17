# Independent URL verification (curl -sL, 2026-10-17 ~20:40 Taipei)
81 unique URLs across lanes. Results: see url_check.txt.
- 73 HTTP 200 directly.
- Verified on retry/alt: proceedings.mlr.press/v139/ceron21a(.html/.pdf) 200 (https); openreview forum?id=OpC-9aBBVJe and forum?id=X6D9bAHhBQ1 200 (pdf links 403 = bot block; cite forum pages); doi.org/10.1109/cig.2014.6932907 -> ieeexplore 6932907 (valid DOI redirect); doi.org/10.1007/978-3-319-71649-7_15 valid DOI redirect.
- DROP (unverifiable / wrong): https://proceedings.mlr.press/v139/obando-ceron21a.html (404; correct is ceron21a.html); http://www.cs.put.poznan.pl/wjaskowski/pub/papers/Szubert2014_2048.pdf (404/timeout; use the DOI instead); http://incompleteideas.net/book/RLbook2020.pdf (unreachable; do not cite URL).
