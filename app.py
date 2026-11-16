"""Gradio demo for Vietnamese aspect-based sentiment analysis."""

import logging

import gradio as gr

from inference import InvalidReview, load_models, predict_review


LOGGER = logging.getLogger(__name__)
HEADERS = ["Khía cạnh", "Cảm xúc"]
EMPTY_STATUS = "Kết quả sẽ xuất hiện sau khi phân tích."
SAMPLES = [
    "Camera chụp ảnh ban ngày rất nét, nhưng pin tụt nhanh khi quay video.",
    "Màn hình sáng và màu đẹp. Máy chạy mượt, nhưng giá hơi cao.",
    "Máy có bộ nhớ 256 GB. Thiết kế gọn, cầm chắc tay và nhân viên tư vấn nhiệt tình.",
    "Loa nghe rõ nhưng máy nóng lên khá nhanh khi chơi game.",
]

CSS = """
.gradio-container {
    --canvas: #f0f4f5;
    --surface: #ffffff;
    --ink: #142b38;
    --subtle: #495f6c;
    --line: #d9e3e7;
    --accent: #0d665c;
    --tint: #e7f2f0;
    background: var(--canvas) !important;
}
.dark .gradio-container, .gradio-container.dark {
    --canvas: #121c25;
    --surface: #1b2934;
    --ink: #eef5f6;
    --subtle: #b7c9d0;
    --line: #384b56;
    --accent: #89decb;
    --tint: #243f43;
}
#workspace {
    max-width: 1120px;
    margin: 0 auto;
    padding: clamp(24px, 4vw, 56px) clamp(4px, 2vw, 20px);
    gap: 18px;
}
#demo-heading {
    padding: 28px clamp(18px, 4vw, 38px);
    border: 1px solid var(--line);
    border-radius: 20px;
    background: linear-gradient(110deg, var(--surface) 38%, var(--tint));
}
#demo-heading .project-name {
    color: var(--accent);
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.11em;
}
#demo-heading h1 {
    margin: 6px 0 8px;
    color: var(--ink);
    font-size: clamp(1.85rem, 4.6vw, 3.1rem);
    font-weight: 700;
    line-height: 1.15;
    letter-spacing: -0.035em;
}
#demo-heading p {
    margin: 0;
    color: var(--subtle);
    font-size: 1.06rem;
}
#analysis-grid { gap: 16px; }
.analysis-card {
    padding: clamp(14px, 2.4vw, 24px);
    border: 1px solid var(--line);
    border-radius: 16px;
    background: var(--surface);
    gap: 14px;
}
.analysis-card h2 { color: var(--ink); font-size: 1.32rem; margin: 0; }
#review textarea { line-height: 1.5; }
#analyze { min-height: 44px; font-weight: 700; }
#result-status {
    color: var(--subtle);
    padding: 10px 14px;
    border-radius: 8px;
    background: var(--tint);
}
#result-status p { margin: 0; }
#review-examples, #model-status {
    padding: 8px 14px;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--surface);
}
@media (max-width: 640px) {
    #analysis-grid > .analysis-card { flex-basis: 100% !important; }
}
"""


def result_table(rows=()):
    return {
        "headers": HEADERS,
        "data": [[aspect, sentiment or "—"] for aspect, sentiment in rows],
    }


def create_demo(predictors, unavailable):
    choices = list(predictors)
    initial_status = EMPTY_STATUS if choices else "Hiện không có mô hình khả dụng."

    def analyze(review, model_name):
        if model_name not in predictors:
            return result_table(), "Vui lòng chọn một mô hình khả dụng."
        try:
            rows = predict_review(review, predictors[model_name])
        except InvalidReview as error:
            return result_table(), str(error)
        except Exception:
            LOGGER.exception("Prediction failed for %s", model_name)
            return result_table(), "Không thể phân tích nhận xét. Vui lòng thử lại."

        if not rows:
            return result_table(), f"**{model_name}** không phát hiện khía cạnh nào."
        message = f"**{model_name}** phát hiện **{len(rows)}** khía cạnh."
        if any(sentiment is None for _, sentiment in rows):
            message += " Dấu — cho biết mô hình chưa gán nhãn cảm xúc."
        return result_table(rows), message

    def reset_results():
        return result_table(), initial_status

    with gr.Blocks(title="Vietnamese ABSA", analytics_enabled=False) as demo:
        with gr.Column(elem_id="workspace"):
            gr.HTML(
                """
                <header>
                  <div class="project-name">VIETNAMESE ABSA</div>
                  <h1>Phân tích cảm xúc theo khía cạnh</h1>
                  <p>Phát hiện khía cạnh (ACD) và phân loại cảm xúc (SC) trong nhận xét về điện thoại.</p>
                </header>
                """,
                elem_id="demo-heading",
            )

            with gr.Row(equal_height=True, elem_id="analysis-grid"):
                with gr.Column(min_width=300, elem_classes="analysis-card"):
                    gr.Markdown("## Nhập nhận xét")
                    review = gr.Textbox(
                        label="Nhận xét tiếng Việt",
                        placeholder="Ví dụ: Camera chụp ảnh đẹp nhưng pin hết nhanh.",
                        lines=5,
                        max_lines=12,
                        elem_id="review",
                    )
                    model = gr.Dropdown(
                        choices=choices,
                        value=choices[0] if choices else None,
                        label="Mô hình",
                        interactive=bool(choices),
                        filterable=False,
                    )
                    submit = gr.Button(
                        "Phân tích", variant="primary", interactive=bool(choices), elem_id="analyze",
                    )

                with gr.Column(min_width=300, elem_classes="analysis-card"):
                    gr.Markdown("## Kết quả phân tích")
                    status = gr.Markdown(initial_status, elem_id="result-status")
                    table = gr.Dataframe(
                        value=result_table(),
                        headers=HEADERS,
                        datatype=["str", "str"],
                        column_count=2,
                        column_widths=["50%", "50%"],
                        interactive=False,
                        show_label=False,
                        buttons=[],
                    )

            gr.Examples(
                examples=[[sample] for sample in SAMPLES],
                inputs=review,
                label="Nhận xét mẫu",
                cache_examples=False,
                elem_id="review-examples",
            )

            availability = f"**Mô hình khả dụng: {len(choices)}/{len(choices) + len(unavailable)}.**"
            if unavailable:
                reasons = "\n".join(f"- **{name}** · {reason}" for name, reason in unavailable.items())
                availability += "\n\nChưa khả dụng\n\n" + reasons
            gr.Markdown(availability, elem_id="model-status")

        submit.click(analyze, [review, model], [table, status], api_name="analyze")
        gr.on(
            [review.change, model.change],
            reset_results,
            outputs=[table, status],
            queue=False,
            api_visibility="private",
        )
    return demo


def main():
    logging.basicConfig(level=logging.INFO)
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Source Sans 3"), "system-ui", "sans-serif"],
    )
    create_demo(*load_models()).queue().launch(share=False, ssr_mode=False, theme=theme, css=CSS)


if __name__ == "__main__":
    main()
