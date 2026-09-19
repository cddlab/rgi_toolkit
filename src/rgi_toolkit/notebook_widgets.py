"""An editable list of native RGI entries for Colab and Jupyter notebooks."""

from __future__ import annotations

import copy
import html
import json

import ipywidgets as W
import yaml
from IPython.display import display

from rgi_toolkit.notebook import FORM_SECTIONS, compose_config, make_config

PENALTIES = ("harmonic", "flat-bottomed", "flat-bottomed1", "flat-bottomed2")
TERMS = {
    "bond": 1,
    "angle": 1,
    "chiral": 1,
    "cistrans": 1,
    "vdw": 1,
    "plane": 0,
    "torsion": 0,
}


def _mapping(text):
    value = yaml.safe_load(text) if text.strip() else {}
    if not isinstance(value, dict):
        raise ValueError("Additional settings must be a YAML/JSON mapping.")
    return value


def _merge(target, extra):
    """Keep advanced fields without silently overriding a visible control."""
    for key, value in extra.items():
        if key in target:
            if isinstance(target[key], dict) and isinstance(value, dict):
                _merge(target[key], value)
            else:
                raise ValueError(f"Set {key!r} in its form field, not twice.")
        else:
            target[key] = copy.deepcopy(value)


def _label(name, control, help_text=""):
    control.description = name
    control.style.description_width = "initial"
    control.layout.width = "100%"
    children = [control]
    if help_text:
        children.append(W.HTML(html.escape(help_text)))
    return W.VBox(children, layout=W.Layout(width="100%"))


class NamedSelections:
    """Add and remove any number of named groups in a custom energy."""

    def __init__(self, selections):
        self.rows = []
        self.container = W.VBox()
        self.add_button = W.Button(description="Add selection", icon="plus")
        self.add_button.on_click(lambda _: self.add())
        self.widget = W.VBox([self.container, self.add_button])
        for name, selection in selections.items():
            self.add(name, selection)

    def add(self, name="", selection=""):
        row = (
            W.Text(value=name, placeholder="Group name, e.g. A"),
            W.Text(value=selection, placeholder="chain A and name CA"),
        )
        remove = W.Button(
            description="Remove selection", icon="trash", layout=W.Layout(width="auto")
        )
        remove.on_click(lambda _: self._remove(row))
        box = W.VBox([_label("name", row[0]), _label("selection", row[1]), remove])
        self.rows.append((row, box))
        self.container.children = tuple(box for _, box in self.rows)

    def _remove(self, row):
        self.rows = [(item, box) for item, box in self.rows if item is not row]
        self.container.children = tuple(box for _, box in self.rows)

    def read(self):
        result = {}
        for (name, selection), _ in self.rows:
            key = name.value.strip()
            if not key or not selection.value.strip():
                raise ValueError("Every custom selection needs a name and a selector.")
            if key in result:
                raise ValueError(f"Duplicate custom selection name: {key}.")
            result[key] = selection.value.strip()
        return result


class RestraintCard:
    """One native entry, with the toolkit's field names and defaults."""

    def __init__(self, editor, kind, config=None, chains="ligands"):
        self.editor, self.kind = editor, kind
        self.fields, self.boxes, self.body = {}, {}, []
        self.base = copy.deepcopy(config or {})
        self.enabled = W.Checkbox(value=True, description="Enabled")
        self.remove = W.Button(description="Remove", icon="trash")
        self.remove.on_click(lambda _: editor.remove(self))
        buttons = [W.HTML(f"<b>{kind}</b>"), self.enabled, self.remove]
        if kind != "conformer":
            self.duplicate = W.Button(description="Duplicate", icon="copy")
            self.duplicate.on_click(self._duplicate)
            buttons.append(self.duplicate)
        if kind in ("distance", "angle"):
            for i, span in enumerate(("1 to 10", "20 to 30", "40 to 50"), 1):
                if kind == "distance" and i == 3:
                    break
                if kind == "distance" and i == 2:
                    span = "40 to 50"
                self._text(f"atom_selection{i}", f"chain A and resid {span}")
            if kind == "angle":
                self.body.append(W.HTML("Group 2 is the vertex of the angle."))
                self._choice("unit", ("degrees", "radians"), "degrees")
        elif kind == "RMSD":
            ref_key = "ref_cif" if "ref_cif" in self.base else "ref_pdb"
            self._choice("reference_format", ("ref_pdb", "ref_cif"), ref_key)
            self._text(
                "reference_file",
                self.base.pop(ref_key, ""),
                "Upload a PDB/mmCIF file using the Files panel.",
            )
            for side in ("target", "ref"):
                self._text(
                    f"atom_selection_{side}",
                    "chain A and name CA" if config is None else "",
                    "Optional shorthand: use these atoms for both fit and calc.",
                )
            selection_start = len(self.body)
            for side in ("target", "ref"):
                for part in ("fit", "calc"):
                    self._text(
                        f"atom_selection_{side}_{part}",
                        "",
                        "Optional override for superposition or RMSD measurement.",
                    )
            self._collapse(selection_start, "Separate fit and calc selections")
            self._choice("pairing", ("align", "identity"), "align")
            self._check("best_effort", True)
        elif kind == "custom":
            self._text("name", f"custom_{editor.serial}")
            source = "use" if "use" in self.base else "energy"
            self._choice("custom_source", ("energy", "use"), source)
            self._text(
                "energy",
                "harmonic(distance(A, B), 25.0)",
                "A toolkit energy expression. Custom angles are in radians.",
            )
            self._text(
                "use", "", "Optional registered function name instead of energy."
            )
            selections = self.base.pop(
                "selections",
                {}
                if config is not None
                else {
                    "A": "chain A and resid 1 to 10",
                    "B": "chain A and resid 40 to 50",
                },
            )
            self.selections = NamedSelections(selections)
            self.body.extend([W.HTML("<b>selections</b>"), self.selections.widget])
        else:
            self._text(
                "conformer_chains",
                chains,
                "Entity opt-in: comma-separated chain IDs, or ligands.",
            )
            self.body.append(
                W.HTML(
                    "One shared conformer configuration applies to all selected chains, "
                    "including protein, DNA, RNA or ligand chains. This is not an atom selector."
                )
            )
            term_start = len(self.body)
            for term, default in TERMS.items():
                native = self.base.pop(term, {}) or {}
                self._number(f"{term}.weight", native.pop("weight", default) or 0)
                if term == "vdw":
                    self._choice(
                        "vdw.mode",
                        ("both", "intramolecular", "intermolecular"),
                        native.pop("mode", "both"),
                    )
                    self._number("vdw.scale", native.pop("scale", 0.75))
                else:
                    unit = {
                        "bond": "Angstrom",
                        "plane": "Angstrom",
                        "chiral": "Angstrom cubed",
                    }.get(term, "radians")
                    self._text(
                        f"{term}.slack",
                        str(native.pop("slack", "")),
                        f"Optional ({unit}); blank uses the toolkit default.",
                    )
                if native:
                    self.base[term] = native
            self._collapse(
                term_start, "Conformer terms: toolkit defaults and overrides"
            )
            self._check("use_esd", False)
            monlib = self.base.pop("monomer_library", None)
            self._text(
                "monomer_library",
                json.dumps(monlib) if monlib is not None else "",
                "Optional YAML: true, a library path, or a mapping.",
            )
        if kind in ("distance", "angle", "RMSD"):
            self._penalty()
        advanced_start = len(self.body)
        if kind != "conformer":
            self._number("weight", 1.0)
            if kind != "RMSD":
                self._text("move", "", "Optional; leave blank for the toolkit default.")
        self._window()
        self.extra = W.Textarea(
            value=yaml.safe_dump(self.base, sort_keys=False), rows=3
        )
        self.body.append(
            _label(
                "Additional entry settings (YAML/JSON)",
                self.extra,
                "Native fields such as refs or detailed VdW settings.",
            )
        )
        self._collapse(
            advanced_start, "Weight, activation window and additional settings"
        )
        self.widget = W.VBox(
            [W.HBox(buttons, layout=W.Layout(flex_flow="row wrap")), *self.body],
            layout=W.Layout(border="1px solid #bbb", padding="12px", margin="8px 0"),
        )
        for key in ("penalty", "window", "custom_source"):
            if key in self.fields:
                self.fields[key].observe(
                    lambda change: self._visibility(), names="value"
                )
        self._visibility()

    def _collapse(self, start, title):
        accordion = W.Accordion([W.VBox(self.body[start:])], selected_index=None)
        accordion.set_title(0, title)
        self.body[start:] = [accordion]

    def _duplicate(self, _):
        config = self.read()
        if self.kind == "custom":
            config["name"] += f"_{self.editor.serial + 1}"
        self.editor.add(self.kind, config)

    def _add(self, name, widget, help_text=""):
        self.fields[name] = widget
        box = _label(name, widget, help_text)
        self.boxes[name] = box
        self.body.append(box)
        return widget

    def _text(self, name, default="", help_text=""):
        value = self.base.pop(name, default)
        if isinstance(value, (dict, list)):
            value = yaml.safe_dump(value, default_flow_style=True).strip()
        return self._add(
            name, W.Text(value="" if value is None else str(value)), help_text
        )

    def _number(self, name, default):
        return self._add(name, W.FloatText(value=float(self.base.pop(name, default))))

    def _choice(self, name, options, default):
        return self._add(
            name, W.Dropdown(options=options, value=self.base.pop(name, default))
        )

    def _check(self, name, default):
        return self._add(
            name, W.Checkbox(value=self.base.pop(name, default), indent=False)
        )

    def _penalty(self):
        kind = next((key for key in PENALTIES if key in self.base), "harmonic")
        params = self.base.pop(kind, {})
        self._choice("penalty", PENALTIES, kind)
        self.quantity = {"RMSD": "rmsd"}.get(self.kind, self.kind)
        target = f"target_{self.quantity}"
        default = {"distance": 25, "angle": 90, "rmsd": 0}[self.quantity]
        for key, value in (
            (target, default),
            (target + "1", max(0, default - 2)),
            (target + "2", default + 2),
        ):
            self._number(key, params.pop(key, value))
        if params:
            raise ValueError(f"Unexpected {kind} parameter(s): {sorted(params)}")

    def _window(self):
        mode = (
            "step"
            if any(k in self.base for k in ("start_step", "stop_step"))
            else (
                "sigma"
                if any(k in self.base for k in ("start_sigma", "stop_sigma"))
                else "always"
            )
        )
        self._choice("window", ("always", "sigma", "step"), mode)
        for key in ("start_sigma", "stop_sigma", "start_step", "stop_step"):
            self._text(key, "", "Optional; blank uses the toolkit default.")

    def _visibility(self):
        def show(name, visible):
            self.boxes[name].layout.display = "" if visible else "none"

        if "penalty" in self.fields:
            penalty = self.fields["penalty"].value
            target = f"target_{self.quantity}"
            show(target, penalty == "harmonic")
            show(target + "1", penalty in ("flat-bottomed", "flat-bottomed1"))
            show(target + "2", penalty in ("flat-bottomed", "flat-bottomed2"))
        for group in ("sigma", "step"):
            for bound in ("start", "stop"):
                show(f"{bound}_{group}", self.fields["window"].value == group)
        if self.kind == "custom":
            for source in ("energy", "use"):
                show(source, self.fields["custom_source"].value == source)

    def read(self):
        values = {key: widget.value for key, widget in self.fields.items()}
        result = {}
        if self.kind in ("distance", "angle"):
            for i in range(1, 3 if self.kind == "distance" else 4):
                result[f"atom_selection{i}"] = values[f"atom_selection{i}"].strip()
            if self.kind == "angle":
                result["unit"] = values["unit"]
        elif self.kind == "RMSD":
            result[values["reference_format"]] = values["reference_file"].strip()
            result.update(pairing=values["pairing"], best_effort=values["best_effort"])
            result.update(
                {
                    key: value.strip()
                    for key, value in values.items()
                    if key.startswith("atom_selection_") and value.strip()
                }
            )
        elif self.kind == "custom":
            source = values["custom_source"]
            result.update(name=values["name"], selections=self.selections.read())
            result[source] = values[source].strip()
        else:
            for term in TERMS:
                result[term] = {"weight": values[f"{term}.weight"]}
                if term == "vdw":
                    result[term].update(
                        mode=values["vdw.mode"], scale=values["vdw.scale"]
                    )
                elif values[f"{term}.slack"].strip():
                    result[term]["slack"] = float(values[f"{term}.slack"])
            result["use_esd"] = values["use_esd"]
            if values["monomer_library"].strip():
                result["monomer_library"] = yaml.safe_load(values["monomer_library"])
        if "penalty" in values:
            penalty, target = values["penalty"], f"target_{self.quantity}"
            suffixes = {
                "harmonic": ("",),
                "flat-bottomed": ("1", "2"),
                "flat-bottomed1": ("1",),
                "flat-bottomed2": ("2",),
            }[penalty]
            result[penalty] = {
                target + suffix: values[target + suffix] for suffix in suffixes
            }
        if "weight" in values:
            result["weight"] = values["weight"]
        if values.get("move", "").strip():
            result["move"] = yaml.safe_load(values["move"])
        mode = values["window"]
        if mode != "always":
            for bound in ("start", "stop"):
                key = f"{bound}_{mode}"
                if values[key].strip():
                    result[key] = (int if mode == "step" else float)(values[key])
        _merge(result, _mapping(self.extra.value))
        return result


class RestraintEditor:
    """Edit any number of distance, angle, RMSD and custom entries together."""

    def __init__(self, chains=(), config=None, conformer_chains="ligands"):
        self.cards, self.serial = [], 0
        self.chain_info, self.status = W.HTML(), W.HTML()
        self.container, self.preview_output = W.VBox(), W.Output()
        self.kind = W.Dropdown(options=tuple(FORM_SECTIONS), description="Type")
        self.add_button = W.Button(description="Add restraint", icon="plus")
        self.add_button.on_click(self._add_clicked)
        self.preview_button = W.Button(
            description="Validate / show config",
            icon="check",
            layout=W.Layout(width="auto", align_self="flex-start"),
        )
        self.preview_button.on_click(self._preview)
        self.settings = W.Textarea(value="verbose: true\n", rows=3)
        self.mode = W.Dropdown(options=("form", "YAML/JSON", "file"), value="form")
        self.config_text = W.Textarea(
            rows=12, placeholder="distance_restraints_config: ..."
        )
        self.config_path = W.Text(placeholder="restraints.yaml")
        self.external_chains = W.Text(value=conformer_chains)
        self.import_button = W.Button(description="Load into form", icon="upload")
        self.import_button.on_click(self._import)
        self.text_box = _label("restraints_config (YAML/JSON)", self.config_text)
        self.path_box = _label(
            "config_path",
            self.config_path,
            "Reference paths are relative to this file.",
        )
        self.external_box = W.VBox(
            [
                self.text_box,
                self.path_box,
                _label(
                    "conformer_chains",
                    self.external_chains,
                    "Entity opt-in for YAML/file mode: chain IDs separated by commas, or ligands.",
                ),
                self.import_button,
            ]
        )
        self.form_box = W.VBox(
            [
                W.HBox([self.kind, self.add_button]),
                self.container,
                _label(
                    "Global settings / other toolkit sections (YAML/JSON)",
                    self.settings,
                ),
            ]
        )
        self.widget = W.VBox(
            [
                self.chain_info,
                W.HTML(
                    "<b>Selections use the RGI-toolkit DSL.</b> Examples: "
                    "<code>chain A and resid 1 to 10 and backbone</code>; "
                    "<code>chain A and (resid 1 to 10 or resid 40 to 50)</code>. "
                    "Residue numbers start at 1 within each chain."
                ),
                _label("Configuration input", self.mode),
                self.status,
                self.form_box,
                self.external_box,
                self.preview_button,
                self.preview_output,
            ]
        )
        self.mode.observe(lambda change: self._source_visibility(), names="value")
        self._source_visibility()
        self.set_chains(chains)
        if config is None:
            self.add("distance")
        else:
            self.load_config(config, conformer_chains=conformer_chains)

    def _source_visibility(self):
        self.form_box.layout.display = "" if self.mode.value == "form" else "none"
        self.external_box.layout.display = "none" if self.mode.value == "form" else ""
        self.text_box.layout.display = "" if self.mode.value == "YAML/JSON" else "none"
        self.path_box.layout.display = "" if self.mode.value == "file" else "none"

    def _import(self, _):
        try:
            config = self.get_config()
            self.load_config(config, conformer_chains=self.external_chains.value)
            self.mode.value = "form"
            self.status.value = "Configuration loaded into the form."
        except (ValueError, TypeError, OSError, yaml.YAMLError) as error:
            self.status.value = html.escape(str(error))

    def set_chains(self, chains):
        text = "; ".join(
            f"{row['chain']}: {row['type']}"
            + (f" ({row['residues']} residues)" if row.get("residues") else "")
            for row in chains
        )
        self.chain_info.value = f"<b>Input chains:</b> {html.escape(text)}"

    def _add_clicked(self, _):
        try:
            self.add(self.kind.value)
            self.status.value = ""
        except ValueError as error:
            self.status.value = f"<b>{html.escape(str(error))}</b>"

    def add(self, kind, config=None, *, chains="ligands"):
        if kind not in FORM_SECTIONS:
            raise ValueError(f"Unknown restraint type: {kind}")
        if kind == "conformer" and any(card.kind == kind for card in self.cards):
            raise ValueError(
                "Edit the existing conformer card and select multiple chains."
            )
        self.serial += 1
        card = RestraintCard(self, kind, config, chains)
        self.cards.append(card)
        self.container.children = tuple(item.widget for item in self.cards)
        return card

    def remove(self, card):
        self.cards.remove(card)
        self.container.children = tuple(item.widget for item in self.cards)

    def load_config(self, config, *, conformer_chains="ligands"):
        config = copy.deepcopy(make_config(config))
        self.cards = []
        for kind, section in FORM_SECTIONS.items():
            data = config.pop(section, None)
            if data is not None:
                for entry in [data] if kind == "conformer" else data:
                    self.add(kind, entry, chains=conformer_chains)
        self.settings.value = yaml.safe_dump(config, sort_keys=False)
        self.container.children = tuple(card.widget for card in self.cards)

    def get_config(self):
        if self.mode.value == "YAML/JSON":
            return make_config(config_text=self.config_text.value)
        if self.mode.value == "file":
            return make_config(config_path=self.config_path.value)
        return compose_config(
            [(card.kind, card.read()) for card in self.cards if card.enabled.value],
            settings=_mapping(self.settings.value),
        )

    def get_conformer_chains(self):
        if self.mode.value != "form":
            return self.external_chains.value
        for card in self.cards:
            if card.kind == "conformer" and card.enabled.value:
                return card.fields["conformer_chains"].value
        return "ligands"

    def _preview(self, _):
        with self.preview_output:
            self.preview_output.clear_output(wait=True)
            try:
                print(yaml.safe_dump(self.get_config(), sort_keys=False))
                print(
                    "Syntax validated. Actual atom matches and counts are checked during setup."
                )
                print("Prediction reads current form values; no Apply step is needed.")
            except (ValueError, TypeError, OSError, yaml.YAMLError) as error:
                print(f"Fix the configuration before prediction: {error}")

    def display(self):
        display(self.widget)


def show_editor(chains, previous=None):
    """Show the same controls again without discarding edits on cell reruns."""
    editor = previous if previous is not None else RestraintEditor(chains)
    editor.set_chains(chains)
    editor.display()
    return editor
