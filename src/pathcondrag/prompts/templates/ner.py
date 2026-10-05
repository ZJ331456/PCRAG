"""Concise source-only NER prompt; the previous prompt is kept separately."""

ner_system = """Extract unique entities named in the paragraph: people, organizations, places, works, languages, dates and quantities. Use source names; keep dates and quantities complete, including units. Never invent names, list generic field labels, or repeat entities.
Return only {"named_entities":["name",...]} with one key. Use an empty list when no entities are named; no explanations."""

one_shot_ner_paragraph = """Radio City
Radio City started in India on 3 July 2001. It plays Hindi and English songs. In May 2008 it launched PlanetRadiocity.com."""

one_shot_ner_output = '{"named_entities":["Radio City","India","3 July 2001","Hindi","English","May 2008","PlanetRadiocity.com"]}'

prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": "${passage}"},
]
