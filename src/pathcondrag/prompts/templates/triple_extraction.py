"""Source-only triples with compact examples of previous failure patterns."""

from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system = """Extract all distinct relationships supported by the paragraph. Return only {"triples":[["subject","relation","object"],...]}: one key, exactly three nonempty strings per triple. Never repeat the entity list, add fields, or use "and" as a relation.
Use the supplied names as hints, not facts. Resolve pronouns only when their referent is clear. Preserve who did what to whom, negation, attribution, dates, units and other qualifiers. Never invent missing names or facts.
Split coordinated subjects/objects only when the same relationship applies to each; respect "respectively". A property's object must come from that same assertion, never the next clause. Use a table cell only when its column is clear; a missing column must not shift neighboring cells or invite guesses. Express properties as complete triples. Deduplicate facts; use compact JSON. Return an empty list only when no relationships are supported."""

ner_conditioned_re_frame = """Paragraph:
{passage}
Names (input only): {named_entity_json}
Return only triples JSON."""

_example_passage = """Riverlight
Nia and Oren are portrayed by Lio. Nia is the daughter and Oren the son of Mara and Vale. Mara calls only Oren "reckless". Lio had classical theater training; Riverlight was meant to look like a historical tale. The Riverlight row lists release date 3 July 2008 and filming location Alder City."""
_example_entities = '{"named_entities":["Riverlight","Nia","Oren","Lio","Mara","Vale","3 July 2008","Alder City"]}'

ner_conditioned_re_input = ner_conditioned_re_frame.format(
    passage=_example_passage, named_entity_json=_example_entities)

ner_conditioned_re_output = '{"triples":[["Nia","portrayed by","Lio"],["Oren","portrayed by","Lio"],["Nia","daughter of","Mara"],["Nia","daughter of","Vale"],["Oren","son of","Mara"],["Oren","son of","Vale"],["Mara","describes Oren as","reckless"],["Lio","trained in","classical theater"],["Riverlight","was meant to look like","a historical tale"],["Riverlight","release date","3 July 2008"],["Riverlight","filming location","Alder City"]]}'

prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    {"role": "user", "content": convert_format_to_template(
        original_string=ner_conditioned_re_frame, placeholder_mapping=None, static_values=None)},
]
