"""An editable list of native RGI entries for Colab and Jupyter notebooks."""

from __future__ import annotations

import copy
import hashlib
import html
import json
from pathlib import Path

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
TYPE_HELP = {
    "distance": "Set the distance between two atom groups (Angstrom).",
    "conformer": "Keep selected chains close to their reference chemical geometry.",
    "angle": "Set the angle between three atom groups; group 2 is the vertex.",
    "custom": "Write an energy formula using named atom groups.",
    "RMSD": "Match a region to a reference PDB or mmCIF structure.",
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
        self.enabled.observe(lambda _: editor._update_summary(), names="value")
        self.remove = W.Button(description="Remove", icon="trash")
        self.remove.on_click(lambda _: editor.remove(self))
        buttons = [W.HTML(f"<b>{kind} #{editor.serial}</b>"), self.enabled, self.remove]
        self.body.append(W.HTML(TYPE_HELP[kind]))
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
                self._text(
                    f"atom_selection{i}",
                    f"chain A and resid {span}",
                    f"Atom group {i}. Example: chain A and resid {span}. "
                    "Use the chain IDs shown above; residue numbering starts at 1.",
                )
            if kind == "angle":
                self.body.append(W.HTML("Group 2 is the vertex of the angle."))
                self._choice("unit", ("degrees", "radians"), "degrees")
        elif kind == "RMSD":
            self.reference_upload = W.FileUpload(
                accept=".pdb,.cif,.mmcif",
                multiple=False,
                description="Upload reference",
                layout=W.Layout(width="auto"),
            )
            self.reference_upload.observe(self._upload_reference, names="value")
            self.body.append(self.reference_upload)
            ref_key = "ref_cif" if "ref_cif" in self.base else "ref_pdb"
            self._choice("reference_format", ("ref_pdb", "ref_cif"), ref_key)
            self._text(
                "reference_file",
                self.base.pop(ref_key, ""),
                "Upload reference fills this in automatically. You can also enter "
                "the path to a file already in this notebook's runtime.",
            )
            for side in ("target", "ref"):
                self._text(
                    f"atom_selection_{side}",
                    "chain A and name CA" if config is None else "",
                    (
                        "Atoms in your prediction."
                        if side == "target"
                        else "Atoms in the uploaded reference."
                    )
                    + " Example: chain A and name CA. Leave empty for the whole structure.",
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
                "Example: harmonic(distance(A, B), 25.0) targets 25 Angstrom between "
                "the groups named A and B below. Custom angles are in radians.",
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
                "Enter chain IDs from the table above, e.g. B,C. "
                "Use ligands to select all ligand chains.",
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
        try:
            config = self.read()
            if self.kind == "custom":
                config["name"] += f"_{self.editor.serial + 1}"
            self.editor.add(self.kind, config)
        except (ValueError, TypeError, yaml.YAMLError) as error:
            self.editor.status.value = html.escape(str(error))

    def _upload_reference(self, change):
        if not change["new"]:
            return
        try:
            upload = change["new"][0]
            name, content = Path(upload["name"]).name, bytes(upload["content"])
            if (
                Path(name).suffix.lower() not in (".pdb", ".cif", ".mmcif")
                or not content
            ):
                raise ValueError("Choose a nonempty PDB or mmCIF reference file.")
            folder = Path(".cache") / "rgi-references"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{hashlib.sha256(content).hexdigest()[:12]}-{name}"
            path.write_bytes(content)
            self.fields["reference_file"].value = str(path)
            self.fields["reference_format"].value = (
                "ref_pdb" if path.suffix.lower() == ".pdb" else "ref_cif"
            )
            self.editor.status.value = f"Reference uploaded: {html.escape(name)}"
        except (ValueError, OSError) as error:
            self.editor.status.value = html.escape(str(error))

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

    def _number(self, name, default, help_text=""):
        return self._add(
            name, W.FloatText(value=float(self.base.pop(name, default))), help_text
        )

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
        meanings = ("exact target", "allowed range", "lower bound", "upper bound")
        self._choice(
            "penalty",
            tuple(
                (f"{key} — {meaning}", key) for key, meaning in zip(PENALTIES, meanings)
            ),
            kind,
        )
        self.quantity = {"RMSD": "rmsd"}.get(self.kind, self.kind)
        target = f"target_{self.quantity}"
        default = {"distance": 25, "angle": 90, "rmsd": 0}[self.quantity]
        unit = "selected angle unit" if self.kind == "angle" else "Angstrom"
        for (key, value), meaning in zip(
            (
                (target, default),
                (target + "1", max(0, default - 2)),
                (target + "2", default + 2),
            ),
            ("Desired value", "Minimum allowed value", "Maximum allowed value"),
        ):
            self._number(key, params.pop(key, value), f"{meaning} ({unit}).")
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
        self.summary = W.HTML()
        self.container, self.preview_output = W.VBox(), W.Output()
        self.add_buttons = {}
        choices = []
        for kind in FORM_SECTIONS:
            button = W.Button(
                description=f"Add {kind}", icon="plus", layout=W.Layout(width="auto")
            )
            button.on_click(lambda _, kind=kind: self._add_clicked(kind))
            self.add_buttons[kind] = button
            choices.append(
                W.VBox(
                    [button, W.HTML(TYPE_HELP[kind])],
                    layout=W.Layout(width="210px", margin="4px 12px 4px 0"),
                )
            )
        self.preview_button = W.Button(
            description="Check RGI settings",
            icon="check",
            layout=W.Layout(width="auto", align_self="flex-start"),
        )
        self.preview_button.on_click(self._preview)
        self.settings = W.Textarea(value="verbose: true\n", rows=3)
        self.mode = W.Dropdown(
            options=(
                ("Edit the form below", "form"),
                ("Paste YAML / JSON", "YAML/JSON"),
                ("Read a YAML / JSON file", "file"),
            ),
            value="form",
        )
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
                self.chain_info,
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
        self.global_settings = W.Accordion(
            [
                _label(
                    "Global settings / other toolkit sections (YAML/JSON)",
                    self.settings,
                )
            ],
            selected_index=None,
        )
        self.global_settings.set_title(0, "Advanced global settings")
        self.native_config = W.Textarea(
            disabled=True, rows=10, layout=W.Layout(width="100%")
        )
        self.native_preview = W.Accordion([self.native_config], selected_index=None)
        self.native_preview.set_title(0, "View the native RGI configuration")
        self.native_preview.layout.display = "none"
        self.form_box = W.VBox(
            [
                W.HTML(
                    "<h4>1. Add the restraints you need</h4>"
                    "Click a button. Repeat to add another restraint of the same type."
                ),
                W.HBox(choices, layout=W.Layout(flex_flow="row wrap")),
                self.summary,
                W.HTML(
                    "<h4>2. Fill in each restraint below</h4>"
                    "Edit the atom groups and target values. "
                    "Duplicate copies an entry; Enabled includes it in prediction."
                ),
                self.chain_info,
                W.HTML(
                    "<b>How to choose atoms:</b> <code>chain A</code> selects a whole chain; "
                    "<code>chain A and resid 1 to 10</code> selects residues 1–10; "
                    "add <code>and name CA</code> for C-alpha atoms only. "
                    "Use the chain IDs in the table. Residue numbers start at 1 in each chain. "
                    "The full RGI-toolkit selection language is also accepted."
                ),
                self.container,
                self.global_settings,
            ]
        )
        self.widget = W.VBox(
            [
                W.HTML(
                    "<h3>Configure RGI</h3>"
                    "Add restraints here, fill in their fields, then check the settings. "
                    "When you are finished, run the next <b>Predict structure</b> cell."
                ),
                _label("Input method", self.mode),
                self.status,
                self.form_box,
                self.external_box,
                W.HTML(
                    "<h4>3. Check, then run prediction</h4>"
                    "Click below to check the settings. Then run the next "
                    "<b>Predict structure</b> cell. Later edits are read automatically."
                ),
                self.preview_button,
                self.preview_output,
                self.native_preview,
            ]
        )
        self.mode.observe(lambda change: self._source_visibility(), names="value")
        self._source_visibility()
        self.set_chains(chains)
        if config is not None:
            self.load_config(config, conformer_chains=conformer_chains)
        self._update_summary()

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
        rows = "".join(
            f"<tr><td style='padding:4px 16px 4px 0'>{html.escape(str(row['chain']))}</td>"
            f"<td style='padding:4px 16px 4px 0'>{html.escape(str(row['type']))}</td>"
            f"<td style='padding:4px 16px 4px 0'>"
            f"{'1–' + str(int(row['residues'])) if row.get('residues') else '—'}</td></tr>"
            for row in chains
        )
        self.chain_info.value = (
            "<b>Your input chains</b><table><thead><tr>"
            "<th style='padding-right:16px;text-align:left'>Chain ID</th>"
            "<th style='padding-right:16px;text-align:left'>Molecule</th>"
            "<th style='text-align:left'>Residue numbers</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            if rows
            else "Enter your molecules in the notebook to see their chain IDs here."
        )

    def _add_clicked(self, kind):
        try:
            self.add(kind)
            self.status.value = f"Added {kind}. Fill in its fields below."
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
        self._update_summary()
        return card

    def remove(self, card):
        self.cards.remove(card)
        self.container.children = tuple(item.widget for item in self.cards)
        self._update_summary()

    def _update_summary(self):
        enabled = [card for card in self.cards if card.enabled.value]
        counts = [
            f"{sum(c.kind == kind for c in enabled)} {kind}"
            for kind in FORM_SECTIONS
            if any(c.kind == kind for c in enabled)
        ]
        self.summary.value = (
            "<b>Enabled:</b> " + "; ".join(counts)
            if counts
            else "<b>No restraints yet.</b> Click Add distance, Add conformer, "
            "Add angle, Add custom or Add RMSD above to begin."
        )

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
        self._update_summary()

    def get_config(self):
        if self.mode.value == "YAML/JSON":
            return make_config(config_text=self.config_text.value)
        if self.mode.value == "file":
            return make_config(config_path=self.config_path.value)
        items = []
        for i, card in enumerate(self.cards, 1):
            if card.enabled.value:
                try:
                    items.append((card.kind, card.read()))
                except (ValueError, TypeError, yaml.YAMLError) as error:
                    raise ValueError(f"{card.kind} entry {i}: {error}") from error
        settings = _mapping(self.settings.value)
        if not items and not any(
            key.endswith("_restraints_config") and value is not None
            for key, value in settings.items()
        ):
            raise ValueError(
                "No RGI restraints are enabled. In the Configure RGI cell, "
                "click Add distance (or another type), fill in its fields, "
                "then run Predict structure. For vanilla, turn use_rgi off."
            )
        return compose_config(items, settings=settings)

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
                config = self.get_config()
                self.native_config.value = yaml.safe_dump(config, sort_keys=False)
                self.native_preview.layout.display = ""
                print("Settings checked. Next: run the Predict structure cell below.")
                print(
                    "Atom availability and reference pairing are checked when prediction starts."
                )
                print("Prediction reads your current edits automatically.")
            except (ValueError, TypeError, OSError, yaml.YAMLError) as error:
                self.native_preview.layout.display = "none"
                print(f"Fix the configuration before prediction: {error}")

    def display(self):
        display(self.widget)


def show_editor(chains, previous=None):
    """Show the same controls again without discarding edits on cell reruns."""
    try:
        from google.colab import output
    except ImportError:
        pass
    else:
        output.enable_custom_widget_manager()
    editor = previous if previous is not None else RestraintEditor(chains)
    editor.set_chains(chains)
    editor.display()
    return editor


def read_editor(editor=None):
    """Give a missing notebook step an actionable error before prediction starts."""
    if editor is None:
        raise ValueError(
            "RGI settings are not open yet. Run the Configure RGI cell with its "
            "left-hand play button, add restraints in the form below it, "
            "then run Predict structure. For vanilla, turn use_rgi off."
        )
    return editor.get_config(), editor.get_conformer_chains()
