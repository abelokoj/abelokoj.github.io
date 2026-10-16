---
layout: page
title: SumatraPDF Enhanced
description: A native Windows document reader extending SumatraPDF with offline dictionaries, vocabulary practice, and customizable annotation tools.
permalink: /projects/sumatrapdf-enhanced/
img: assets/img/projects/sumatrapdf-enhanced.png
importance: 2
category: software
giscus_comments: false
---

SumatraPDF Enhanced is an open-source Windows application I am developing to bring document reading, annotation, and vocabulary study into one environment. It builds on [SumatraPDF](https://github.com/sumatrapdfreader/sumatrapdf), retaining its document-rendering and navigation foundation while adding tools for studying technical material and working with PDF documents.

The project combines native C++ and Win32 interface development with PDF annotation handling, offline dictionary indexing, and vocabulary-review scheduling. Its design emphasizes a compact reading interface, configurable tools, and offline access to definitions and learning activities.

[View the source code](https://github.com/abelokoj/sumatrapdf) · [Read the project documentation](https://github.com/abelokoj/sumatrapdf#readme) · [Report an issue](https://github.com/abelokoj/sumatrapdf/issues)

## Reading and vocabulary study

- **Offline dictionary lookup.** Select a word and press **Shift+D** to look up its meaning. The application supports bundled study dictionaries, WordNet coverage, dictionary imports, and optional additional dictionary packs.
- **Vocabulary with document context.** Save words with definitions, source documents, page numbers, and reading context; organize custom lists and import or export vocabulary backups.
- **Six learning activities.** Practice with flashcards, meaning quizzes, word quizzes, spelling, word scramble, and matching pairs. SM-2 and Leitner review scheduling support repeated study.
- **Pronunciation.** Use installed Windows voices offline, with optional online pronunciation recordings when explicitly requested.

## Annotation and presentation

The annotation tools extend the reader's native PDF annotation support with ballpoint, fountain, brush, pencil, and highlighter profiles. Pen settings include independent colors, widths, and opacity, with favorite presets that can be pinned to the toolbar.

Current development also includes lasso selection for moving, resizing, and deleting annotations; an explicit hand tool for page navigation; and a temporary laser pointer with solid-trail, hollow-trail, and dot styles. Annotation saving preserves native PDF data, while the laser provides temporary presentation marks.

Highlight and note exports support Markdown, plain text, Typst, HTML, Word, JSON, and CSV, including links back to source pages.

## Interface and implementation

The interface provides theme presets, bundled font choices, adjustable text and icon sizes, and controls that adapt to interface scaling and monitor DPI. Recent-document navigation, toolbar overflow, and settings layout are being refined alongside the reading and learning tools.

The application uses SumatraPDF's MuPDF-based rendering foundation. Development includes native regression checks for interface geometry, settings persistence, annotation behavior, and document loading. GitHub Actions workflows build Windows x64 and ARM64 packages.

## Project status and attribution

The project is under active development. This overview includes features in the current development source; availability in packaged releases may differ. The repository documentation and changelog describe the release-specific feature set. Advanced handwriting recognition and handwriting cleanup remain future work.

SumatraPDF Enhanced is derived from the original SumatraPDF project, with visual inspiration from [PrettySumatraPDF](https://github.com/JaviLendi/PrettySumatraPDF) and vocabulary resources credited to [WMKeyboard](https://github.com/wasi-master/wmkeyboard). Document rendering, text search, native annotations, document inversion, and reference previews are retained upstream capabilities that the project extends. Application and bundled-resource licensing are documented in the [repository](https://github.com/abelokoj/sumatrapdf#readme).
