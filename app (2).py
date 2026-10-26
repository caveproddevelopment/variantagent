"""Gradio UI for the standalone Job Title Variants Agent (HF Spaces / Railway)."""

import json
import os
import tempfile

import gradio as gr

from variants_agent import N_VARIANTS, generate_for_titles, to_rows, write_csv


def run(t1, t2, t3):
    titles = [t for t in (t1, t2, t3) if t and t.strip()]
    try:
        results = generate_for_titles(titles)
    except ValueError as e:
        return f"**Error:** {e}", None, None, None
    except Exception as e:  # API/network errors
        return f"**Agent failed:** {type(e).__name__}: {e}", None, None, None

    lines = []
    for r in results:
        line = f"**{r['input_title']}** — {r['count']} variants"
        if r["warning"]:
            line += f"  \n⚠️ {r['warning']}"
        lines.append(line)

    rows = to_rows(results)
    table = [[x["input_title"], x["rank"], x["variant"], x["closeness"]] for x in rows]

    out_dir = tempfile.mkdtemp()
    csv_path = os.path.join(out_dir, "title_variants.csv")
    write_csv(results, csv_path)

    return "\n\n".join(lines), table, csv_path, json.dumps(results, indent=2)


with gr.Blocks(title="Job Title Variants Agent") as demo:
    gr.Markdown(f"# Job Title Variants Agent\nEnter 1–3 job titles. Each returns {N_VARIANTS} ranked variants.")

    with gr.Row():
        with gr.Column(scale=1):
            t1 = gr.Textbox(label="Job title 1", placeholder="e.g. Data Engineer")
            t2 = gr.Textbox(label="Job title 2 (optional)")
            t3 = gr.Textbox(label="Job title 3 (optional)")
            btn = gr.Button("Generate variants", variant="primary")
        with gr.Column(scale=2):
            status = gr.Markdown("")
            table = gr.Dataframe(
                headers=["input_title", "rank", "variant", "closeness"],
                label="Variants", wrap=True, interactive=False,
            )
            csv_file = gr.File(label="CSV download")
            json_out = gr.Code(label="JSON", language="json")

    btn.click(run, inputs=[t1, t2, t3], outputs=[status, table, csv_file, json_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
