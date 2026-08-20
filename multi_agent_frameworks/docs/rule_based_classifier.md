# Rule-Based Suicide-Related Post Classifier

## Purpose and scope

`rule_based` is a deterministic baseline for the repository's five-label research task. It uses regular expressions and local context rules; it does not call an LLM and is not a clinical screening, triage, or intervention tool. It returns the same evaluator-facing final output as every workflow:

```json
{"label_name": "Ideation", "suicidal_intent": true}
```

`suicidal_intent` is derived, not independently predicted: it is `false` for `Unsupportive` and `true` for `Indicator`, `Ideation`, `Behavior`, and `Attempt`.

## Research basis

The implementation is deliberately a context-aware rule system instead of a simple keyword list.

- The C-SSRS differentiates suicidal ideation, preparatory acts, and actual attempts. An actual attempt is a potentially self-injurious act accompanied by at least some wish to die; preparatory acts include actions beyond thoughts, such as preparing a method or a goodbye note. [C-SSRS definitions](https://cssrs.columbia.edu/wp-content/uploads/C-SSRS_ChildBaseline_11.14.16.pdf)
- The NIMH ASQ includes wishing to be dead, believing others would be better off if one were dead, thoughts of killing oneself, and a prior attempt. This motivates the passive- and active-ideation phrase families. [NIMH ASQ](https://www.nimh.nih.gov/sites/default/files/documents/research/research-conducted-at-nimh/asq-toolkit-materials/asq-tool/screening-tool-asq-nimh-toolkit.pdf)
- Published rule-based suicide NLP systems combine target lexicons with modifiers for negation, historical context, conditional wording, and another person/family context; they do not label every occurrence of a suicide term as current personal ideation. [Current ideation rule system](https://pmc.ncbi.nlm.nih.gov/articles/PMC8009838/), [multi-site context study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10980141/)

The implementation therefore uses text normalization, sentence-local matching, self-versus-other context, explicit denials, benign idiom exclusions, and label precedence.

## Glossary and rules

| Label | Meaning in this repository | Rule families used |
|---|---|---|
| `Attempt` | The author reports a current or past suicide attempt. | First-person patterns such as `I attempted suicide`, `I tried to kill myself`, `my suicide attempt`, and attempt aftermath such as `survived my attempt`. |
| `Behavior` | The author reports preparation or self-inflicted harm beyond thoughts alone. | First-person suicide plans, writing a goodbye/suicide note, preparing for death, giving possessions away in suicide context, and explicit self-harm such as `I cut myself`, `I hurt myself on purpose`, or `I burned myself`. |
| `Ideation` | The author reports wishing to die or thinking about ending their life, without the above action evidence. | First-person passive wishes (`wish I were dead`, `wish I would not wake up`, `better off dead`) and active phrases (`I am suicidal`, `thinking about suicide`, `kill myself`, `end my life`, `do not want to live`). Hypothetical self-directed suicide wording remains ideation when no action is present. |
| `Indicator` | Suicide/self-harm is meaningfully discussed but the author does not positively assert their own ideation, behavior, or attempt. | Suicide/self-harm anchors in third-person, general, or news/support discussion; explicit personal denials such as `I am not suicidal` or `I do not want to kill myself`. |
| `Unsupportive` | The post does not meaningfully concern suicide or self-harm. | Fallback after all rules. Explicit benign expressions, including `suicide squad`, `career suicide`, and `dead tired`, are ignored rather than promoted to `Indicator`. Depression, sadness, or anger alone are not positive evidence. |

## Precedence and context

The classifier evaluates all sentences, then chooses the highest matched category:

```text
Attempt > Behavior > Ideation > Indicator > Unsupportive
```

This prevents a post that contains both `I tried to kill myself` and `I want to die` from being reduced to Ideation. The classifier does not infer an author-level positive label solely from a suicide mention by another person, a family-history statement, or an explicit self-denial. A past first-person attempt remains `Attempt`, matching the repository's label definition.

The implementation is intentionally conservative about action claims: a generic word such as `attempt`, `plan`, `cut`, or `overdose` is insufficient without self-directed and suicide/self-harm context. The rules do not score depression, hopelessness, or distress as suicidality by themselves.

## Implementation plan

1. Normalize Unicode apostrophes and whitespace, then inspect sentence-local text.
2. Apply benign-idiom filtering, personal denial rules, and third-person context handling.
3. Gather matches for the four positive phases and choose the highest-precedence phase.
4. Return `Unsupportive` only if no meaningful suicide/self-harm evidence remains.
5. Store the selected rule and matched sentence in `agent_outputs` and a trace event for auditability.
6. Register `rule_based` with the existing CLI and shared evaluator. It produces zero LLM calls; model, service-tier, token, and retry flags are accepted by the shared CLI but have no effect.
7. Cover representative patterns, exclusions, precedence, CLI execution, and report mapping with tests.

## Limitations

Regex rules cannot reliably resolve sarcasm, implied intent, unusual phrasing, multiple speakers, or nuanced clinical context. The source studies emphasize iterative development and validation against a labeled corpus; this implementation is a transparent baseline, not a validated clinical instrument. It must not be used for patient care or automated safety action.
