## [v0.7.0](https://github.com/docling-project/docling-agent/releases/tag/v0.7.0) - 2026-10-21

### Feature

* Depend on docling-slim and convert sources locally or with a Docling service ([#59](https://github.com/docling-project/docling-agent/issues/59)) ([`c3e47bb`](https://github.com/docling-project/docling-agent/commit/c3e47bb1b05a398b613e7e21e7509ac07ef852ca))
* **perfs:** Add VLM-based picture/chart extraction to RAG eval pipeline ([#57](https://github.com/docling-project/docling-agent/issues/57)) ([`9c754cf`](https://github.com/docling-project/docling-agent/commit/9c754cf9b581e082e7ac2c45f3638223562f7578))
* **rag:** Add ViDoRe V3 agentic RAG evaluation with reasoning-based page selectors ([#43](https://github.com/docling-project/docling-agent/issues/43)) ([`6275d15`](https://github.com/docling-project/docling-agent/commit/6275d152b38029a0fe2f99a0e947be3dc592b7b6))
* **trace:** Generalize the RAG trace to all agents and export it to a file ([#42](https://github.com/docling-project/docling-agent/issues/42)) ([`f80718e`](https://github.com/docling-project/docling-agent/commit/f80718e28ae5744fde5b3239b2046ed7c2b61580))
* Make doclang the default container for library sources ([#49](https://github.com/docling-project/docling-agent/issues/49)) ([`675cc60`](https://github.com/docling-project/docling-agent/commit/675cc60363cd29d4de356f5db3338f34fc4cbc43))
* **examples:** Page-by-page entity extraction with multi-model comparison ([#35](https://github.com/docling-project/docling-agent/issues/35)) ([`26d95cd`](https://github.com/docling-project/docling-agent/commit/26d95cd3b9aec7db213777fd1063311a1c00cece))

### Fix

* List tables and text items in the editor outline ([#58](https://github.com/docling-project/docling-agent/issues/58)) ([`0a91a95`](https://github.com/docling-project/docling-agent/commit/0a91a957adfb6cdd65080676723649a891f04bdd))
* Bugs and improvements after library refactoring ([#55](https://github.com/docling-project/docling-agent/issues/55)) ([`050161c`](https://github.com/docling-project/docling-agent/commit/050161c008de0e94cc736f0e35bcea2cace9054a))
* **backends:** Support URL query parameters ([#48](https://github.com/docling-project/docling-agent/issues/48)) ([`caae887`](https://github.com/docling-project/docling-agent/commit/caae887738e6aad8009fe1655b52fcf718cb28cc))
* **editor:** Remove stale items after rewrite ([#46](https://github.com/docling-project/docling-agent/issues/46)) ([`e3de427`](https://github.com/docling-project/docling-agent/commit/e3de427dc3ce09099672fad0dc25b74d0e8cb7b9))

### Documentation

* Improve README clarity, consistency, links ([#56](https://github.com/docling-project/docling-agent/issues/56)) ([`8037080`](https://github.com/docling-project/docling-agent/commit/803708031ac0598150dcb44a37ca3faa572bfe03))

## [v0.6.0](https://github.com/docling-project/docling-agent/releases/tag/v0.6.0) - 2026-06-25

### Feature

* **rag:** Expose reasoning trace via public run_with_trace() ([#39](https://github.com/docling-project/docling-agent/issues/39)) ([`9075749`](https://github.com/docling-project/docling-agent/commit/9075749064eb369e0e72d7cd218598d6e4ad2381))

### Fix

* **enrich:** Propagate task to enricher and enforce label whitelist ([#31](https://github.com/docling-project/docling-agent/issues/31)) ([`33cd343`](https://github.com/docling-project/docling-agent/commit/33cd3436f995f38cf90e6453aa854012bdf08f78))

## [v0.5.0](https://github.com/docling-project/docling-agent/releases/tag/v0.5.0) - 2026-05-29

### Feature

* **eval:** Add extraction evaluation utility ([#22](https://github.com/docling-project/docling-agent/issues/22)) ([`ac68801`](https://github.com/docling-project/docling-agent/commit/ac688010f1f18bb65d173f05307339540b232124))

### Fix

* Linting on 'perfs' folder ([#33](https://github.com/docling-project/docling-agent/issues/33)) ([`ee2259e`](https://github.com/docling-project/docling-agent/commit/ee2259edeaee951d7e2edfd8905311d5f354dadf))

## [v0.4.0](https://github.com/docling-project/docling-agent/releases/tag/v0.4.0) - 2026-05-29

### Feature

* **enricher:** Add page-level summarization ([#32](https://github.com/docling-project/docling-agent/issues/32)) ([`2d88ad2`](https://github.com/docling-project/docling-agent/commit/2d88ad2068e30afad0e99022496d743c8f012add))

## [v0.3.0](https://github.com/docling-project/docling-agent/releases/tag/v0.3.0) - 2026-05-22

### Feature

* Added the llama-server backend ([#28](https://github.com/docling-project/docling-agent/issues/28)) ([`cb2ad4c`](https://github.com/docling-project/docling-agent/commit/cb2ad4ced04efa1784297b4599a36a0b431ba031))

### Fix

* **enricher:** Use the right field names of EntityMention ([#29](https://github.com/docling-project/docling-agent/issues/29)) ([`864445e`](https://github.com/docling-project/docling-agent/commit/864445ebf58d1099d365776643ec81dbce82009b))

## [v0.2.0](https://github.com/docling-project/docling-agent/releases/tag/v0.2.0) - 2026-05-20

### Feature

* Add new backends to agent ([#27](https://github.com/docling-project/docling-agent/issues/27)) ([`3ef9f7f`](https://github.com/docling-project/docling-agent/commit/3ef9f7f1b88ba5f2b03992e16bc4743ece381098))

## [v0.1.0](https://github.com/docling-project/docling-agent/releases/tag/v0.1.0) - 2026-04-10

### Feature

* Add enrichment agent ([#7](https://github.com/docling-project/docling-agent/issues/7)) ([`e27b201`](https://github.com/docling-project/docling-agent/commit/e27b2019f0815b9d5a8b5d21dc5a0a13ee21ceea))
* Added different models for writing and planning, added the captioning of the table ([#6](https://github.com/docling-project/docling-agent/issues/6)) ([`7df65c3`](https://github.com/docling-project/docling-agent/commit/7df65c3bad45e1a5ca3a5503338f9115528e5ef3))

### Fix

* Improve DoclingEnrichingAgent heading level fixes and related enhancements ([#14](https://github.com/docling-project/docling-agent/issues/14)) ([`b359960`](https://github.com/docling-project/docling-agent/commit/b359960d61e3bb54019115e0a7f66f728e570a27))
* Update mellea due to API changes and lock the version ([#10](https://github.com/docling-project/docling-agent/issues/10)) ([`2511c3f`](https://github.com/docling-project/docling-agent/commit/2511c3f9a62f6496078ed804f43960560f671bb1))
