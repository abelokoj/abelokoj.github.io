# MLModern offline fonts

Downloaded from [abelokoj/mlmodern-otf-ttf](https://github.com/abelokoj/mlmodern-otf-ttf/tree/0bb1ce317a4f0d438503ad4b2d71913eccaa1555) at revision `0bb1ce317a4f0d438503ad4b2d71913eccaa1555` on October 15, 2026.

## Text

`otf/` contains the unmodified MLModern Regular, Italic, Bold, and Bold Italic files. `woff2/` contains losslessly compressed webfont versions made with fontTools 4.60.1 and Brotli; these are complete fonts, not character subsets. Outlines, character coverage, kerning, ligatures, and copyright metadata are retained. The website uses these four faces for prose and headings. The résumé keeps its existing Open Sans styling, and code keeps its Maple Mono NF fallback chain. Existing font sizes and spacing are unchanged.

These text fonts derive from MLModern 1.2 by Daniel Benjamin Miller (2021), based on Latin Modern by Bogusław Jackowski and Janusz M. Nowacki. They are licensed under the LaTeX Project Public License 1.3c; see `LPPL-1.3c.txt` and the upstream `MANIFEST-mlmodern.txt`.

## Mathematics

The original `MLModernMath-Regular.otf` and `MLModernMath-Bold.otf`, and compressed WOFF2 copies, are also retained. As documented by the font repository, MLModern Math is a renamed New Computer Modern Math 8.1.1: Regular uses its Book face, and Bold uses its Bold face. These fonts include OpenType MATH tables. They are not the separate, incomplete MLModern Math Letters/Symbols/Extension faces.

New Computer Modern Math is by Antonis Tsolomitis (2019–2026). Its math faces use the GUST Font License 1.0, governed by LPPL 1.3c. See `GUST-FONT-LICENSE.txt`, `LPPL-1.3c.txt`, `NewCM-License.txt`, and `MANIFEST-mlmodern.txt`. The GPL exceptions discussed in `NewCM-License.txt` concern other New Computer Modern faces, not these math files.

### Website renderer

MathJax SVG does not typeset directly from an arbitrary OTF or WOFF2 file. `mathjax/tex-mml-svg-mlmodern.min.js` is therefore a site-specific MathJax 4.1.0 build containing SVG paths and metrics extracted from these pinned OTFs. Ordinary text within equations uses MLModern; mathematical alphabets, Greek, symbols, operators, and extensible constructions come from MLModern Math Regular. The Bold math file is retained for offline use; mathematical bold alphabets on the website use the regular math face's dedicated bold glyphs.

The conversion uses [dgleich/mathjax-fonts](https://github.com/dgleich/mathjax-fonts/tree/34075ed04fe7db9153772948c2b1e0938fcd63a0), revision `34075ed04fe7db9153772948c2b1e0938fcd63a0` (MIT-licensed tooling), with fontTools 4.60.1. The font's MATH table provides fraction/radical rule widths; the text font provides its actual x-height. Identical mathematical glyph data in inherited variants is stored only once. No glyph outlines are redesigned.

The renderer is bundled with webpack 5.102.1, webpack-cli 6.0.1, and terser-webpack-plugin 5.3.14, using `@mathjax/src` 4.1.0 with `@xmldom/xmldom` overridden to patched version 0.9.12. It keeps MathJax's normal accessibility support, assistive MathML, TeX/MathML input, and equation numbering; it does not use the converter's placeholder speech worker. Optional MathJax extensions and speech resources resolve against the pinned `https://cdn.jsdelivr.net/npm/mathjax@4.1.0` distribution. MathJax and Speech Rule Engine use Apache-2.0 (`mathjax/MathJax-LICENSE.txt`).

The build tools and intermediate data stay in the site's excluded `_private/mlmodern-web-build/` directory. There is no additional npm installation or font build step in GitHub Pages deployment. `_config.yml` points to the ready-made renderer and validates it with SHA-256 integrity.

The older Libertinus and Splentinex font folders are intentionally kept for offline use.
