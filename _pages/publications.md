---
layout: page
permalink: /publications/
title: Publications
description: Peer-reviewed articles and preprints, most recent first.
scholar:
  sort_by: year
  order: descending
nav: true
nav_order: 6
---

<!-- _pages/publications.md -->

<!-- Bibsearch Feature -->

{% include bib_search.liquid %}

<div class="publications">

{% bibliography --file papers --template bib %}

</div>
