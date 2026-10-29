# Antbdy-Antiquity
antbdy-antiquity is a web-based computational workbench that unifies the antibody lifecycle. Rather than designing synthetic antibodies in a vacuum or guessing mutations blindly in the lab, it draws on how nature evolves antibodies against real diseases and uses published AI models to recommend candidate mutations.

Current prototype pipeline:

Input: an antibody sequence
Natural-mutation prior: ThriftySHM / S5F (via netam) scores how likely somatic hypermutation is to produce each mutation
Plausibility ranking: AbLang2 ranks candidates by antibody-language-model plausibility
Structure view: IgFold predicts structures for the top-ranked mutations
Status: research prototype. Experimental validation (e.g., against the gcreplay dataset) is not yet included, and outputs are computational suggestions, not confirmed improvements.
