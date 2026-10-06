# AccidentIQ — Phase 8–9 modality assessment

## Decision

No image model or tabular–image fusion was trained. The available public image/video datasets cannot be linked to AccidentIQ's 2019–2021 STATS19 collision rows with a shared collision identifier or defensible event-level mapping. Joining by approximate time, road context, geography or severity would create unverified labels and leakage risk.

## Evidence reviewed

- The [DfT Road Safety Open Data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data) describes collision, vehicle and casualty records from police-reported personal-injury collisions. It provides no associated scene images or image identifier in the released collision tables.
- The [AI City Challenge 2024 Traffic Safety Description and Analysis data](https://www.aicitychallenge.org/2024-data-and-evaluation/) includes video segments and annotations for staged scenarios. It has its own scenario/segment labels, not STATS19 collision keys, and its controlled staged events do not represent the same population or target as real reported GB collisions.
- [Cityscapes](https://www.cityscapes-dataset.com/dataset-overview/) contains street-scene imagery and GPS/vehicle metadata, but its labels describe visual scene semantics, not STATS19 accident events or injury severity.

## Phase status

**Phase 8 completed as a dataset suitability review:** no transfer-learning image classifier was justified by the available data.

**Phase 9 completed as a documented no-go:** no fusion model was constructed because there is no trustworthy paired sample. Independent datasets remain separate; no synthetic pairing is permitted.

Reconsider this only when image/video records can be joined to collision records using a documented shared event key or an auditable camera/time/location linkage, with permissions, timestamp tolerances, duplicate-event handling and a split strategy that keeps the same event/camera out of train and evaluation. Until then, the tabular model is the supported system.
