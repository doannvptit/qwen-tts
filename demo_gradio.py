import argparse

import gradio as gr
import torch

from src.model import LlmSpokenModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--system-prompt", type=str, default="Say exactly provided sentence."
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def pick_device(raw_device: str) -> torch.device:
    if raw_device == "cpu":
        return torch.device("cpu")
    if raw_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_app(model: LlmSpokenModel, system_prompt: str):
    def init_messages():
        return [{"role": "system", "content": system_prompt}]

    def push_user_message(user_text, chat_history, messages):
        text = (user_text or "").strip()
        if not text:
            raise gr.Error("Please enter a message.")

        next_messages = list(messages)
        next_messages.append({"role": "user", "content": text})

        next_chat = list(chat_history or [])
        next_chat.append({"role": "user", "content": text})
        next_chat.append({"role": "assistant", "content": ""})
        return "", next_chat, next_messages, None, "Thinking..."

    def stream_assistant(chat_history, messages, max_new_tokens, temperature, top_p):
        try:
            final_payload = None
            print(messages)
            for payload in model.stream_generate_assistant(
                messages=messages,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
            ):
                if payload["event"] == "text":
                    next_chat = list(chat_history)
                    next_chat[-1] = {"role": "assistant", "content": payload["text"]}
                    yield next_chat, messages, None, "Thinking..."
                elif payload["event"] == "final":
                    final_payload = payload

            if final_payload is None:
                raise gr.Error("Generation ended without a final payload.")

            final_text = str(final_payload["text"])
            next_messages = list(messages)
            next_messages.append({"role": "assistant", "content": final_text})
            next_chat = list(chat_history)
            next_chat[-1] = {"role": "assistant", "content": final_text}

            audio = final_payload["audio"]
            if audio is None:
                yield next_chat, next_messages, None, "Done (text only)"
            else:
                sample_rate = int(final_payload["sample_rate"])
                yield next_chat, next_messages, (sample_rate, audio), "Done"
        except Exception as exc:
            next_chat = list(chat_history)
            fallback_text = "I could not generate a response. Please try again."
            next_chat[-1] = {"role": "assistant", "content": fallback_text}
            next_messages = list(messages)
            next_messages.append({"role": "assistant", "content": fallback_text})
            yield next_chat, next_messages, None, f"Generation failed: {exc}"

    def clear_chat():
        return [], init_messages(), None, "Ready"

    def regenerate(chat_history, messages):
        if not chat_history:
            raise gr.Error("No previous turn to regenerate.")

        trimmed_messages = list(messages)
        while trimmed_messages and trimmed_messages[-1]["role"] == "assistant":
            trimmed_messages.pop()
        if not trimmed_messages or trimmed_messages[-1]["role"] != "user":
            raise gr.Error(
                "Regenerate expects the previous turn to end with a user message."
            )

        trimmed_chat = list(chat_history)
        if trimmed_chat and trimmed_chat[-1]["role"] == "assistant":
            trimmed_chat.pop()
        trimmed_chat.append({"role": "assistant", "content": ""})
        return trimmed_chat, trimmed_messages, None, "Thinking..."

    with gr.Blocks(title="LLM Spoken Chat") as demo:
        gr.Markdown("# LLM Spoken Demo")
        gr.Markdown(
            "Multi-turn chat that streams text first, then returns synthesized speech."
        )

        messages_state = gr.State(init_messages())
        status = gr.Markdown("Ready")

        chatbot = gr.Chatbot(height=560)
        with gr.Row():
            msg = gr.Textbox(
                label="Message",
                placeholder="Ask anything...",
                lines=2,
                scale=8,
            )
            send = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            regen = gr.Button("Regenerate")
            clear = gr.Button("Clear")

        with gr.Accordion("Generation settings", open=False):
            max_new_tokens = gr.Slider(
                32, 512, value=220, step=1, label="Max new tokens"
            )
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")

        assistant_audio = gr.Audio(label="Assistant audio", autoplay=True)

        submit_event = msg.submit(
            push_user_message,
            inputs=[msg, chatbot, messages_state],
            outputs=[msg, chatbot, messages_state, assistant_audio, status],
        )
        submit_event.then(
            stream_assistant,
            inputs=[chatbot, messages_state, max_new_tokens, temperature, top_p],
            outputs=[chatbot, messages_state, assistant_audio, status],
        )

        click_event = send.click(
            push_user_message,
            inputs=[msg, chatbot, messages_state],
            outputs=[msg, chatbot, messages_state, assistant_audio, status],
        )
        click_event.then(
            stream_assistant,
            inputs=[chatbot, messages_state, max_new_tokens, temperature, top_p],
            outputs=[chatbot, messages_state, assistant_audio, status],
        )

        regen_event = regen.click(
            regenerate,
            inputs=[chatbot, messages_state],
            outputs=[chatbot, messages_state, assistant_audio, status],
        )
        regen_event.then(
            stream_assistant,
            inputs=[chatbot, messages_state, max_new_tokens, temperature, top_p],
            outputs=[chatbot, messages_state, assistant_audio, status],
        )

        clear.click(
            clear_chat,
            inputs=None,
            outputs=[chatbot, messages_state, assistant_audio, status],
        )

    return demo


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    model = LlmSpokenModel.from_pretrained(args.checkpoint)
    model.to(device)
    model.model.eval()
    model.talker.eval()
    model.wav_encoder.eval()
    model.wav_decoder.eval()

    app = create_app(model=model, system_prompt=args.system_prompt)
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
