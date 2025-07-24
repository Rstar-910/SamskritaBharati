import gradio as gr
import torch
from unsloth import FastLanguageModel
from IPython.display import display, Audio
import numpy as np

model = None
tokenizer = None
snac_model = None
device = None

def load_models():
    """Load models once when the app starts"""
    global model, tokenizer, snac_model, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models on: {device}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        "rverma0631/lora_model_sanskrit_tts",  # Your pushed model
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )
    
    # Prepare model for inference
    model = model.to(device)
    FastLanguageModel.for_inference(model)
    
    try:
        from snac import SNAC
        snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    except ImportError:
        print("Warning: SNAC model import failed. Make sure SNAC is installed.")

    snac_model.to("cpu")
    
    print("Models loaded successfully!")

def redistribute_codes(code_list):
    """Your original redistribute_codes function"""
    layer_1 = []
    layer_2 = []
    layer_3 = []
    
    for i in range((len(code_list)+1)//7):
        layer_1.append(code_list[7*i])
        layer_2.append(code_list[7*i+1]-4096)
        layer_3.append(code_list[7*i+2]-(2*4096))
        layer_3.append(code_list[7*i+3]-(3*4096))
        layer_2.append(code_list[7*i+4]-(4*4096))
        layer_3.append(code_list[7*i+5]-(5*4096))
        layer_3.append(code_list[7*i+6]-(6*4096))
    
    codes = [torch.tensor(layer_1).unsqueeze(0),
             torch.tensor(layer_2).unsqueeze(0),
             torch.tensor(layer_3).unsqueeze(0)]
    
    # codes = [c.to("cuda") for c in codes]
    audio_hat = snac_model.decode(codes)
    return audio_hat

def sanskrit_tts_inference(sanskrit_text, chosen_voice=""):
    """
    Your original inference pipeline adapted for Gradio
    """
    if not sanskrit_text.strip():
        return None, "Please enter some Sanskrit text."
    
    try:
        # Process prompts (single prompt in this case)
        prompts = [sanskrit_text]
        chosen_voice = chosen_voice if chosen_voice.strip() else None
        
        prompts_ = [(f"{chosen_voice}: " + p) if chosen_voice else p for p in prompts]
        
        all_input_ids = []
        for prompt in prompts_:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            all_input_ids.append(input_ids)
        
        start_token = torch.tensor([[ 128259]], dtype=torch.int64) # Start of human
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64) # End of text, End of human
        
        all_modified_input_ids = []
        for input_ids in all_input_ids:
            modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1) # SOH SOT Text EOT EOH
            all_modified_input_ids.append(modified_input_ids)
        
        all_padded_tensors = []
        all_attention_masks = []
        max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])
        
        for modified_input_ids in all_modified_input_ids:
            padding = max_length - modified_input_ids.shape[1]
            padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
            attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
            all_padded_tensors.append(padded_tensor)
            all_attention_masks.append(attention_mask)
        
        all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)
        
        # Use device variable instead of hardcoded "cuda"
        input_ids = all_padded_tensors.to(device)
        attention_mask = all_attention_masks.to(device)
        
        # Generate using your original parameters
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1200,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            num_return_sequences=1,
            eos_token_id=128258,
            use_cache=True
        )
        
        # Your original post-processing
        token_to_find = 128257
        token_to_remove = 128258
        
        token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)
        
        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
        else:
            cropped_tensor = generated_ids
        
        mask = cropped_tensor != token_to_remove
        
        processed_rows = []
        for row in cropped_tensor:
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)
        
        code_lists = []
        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            trimmed_row = [t - 128266 for t in trimmed_row]
            code_lists.append(trimmed_row)
        
        # Generate audio using your original function
        my_samples = []
        for code_list in code_lists:
            samples = redistribute_codes(code_list)
            my_samples.append(samples)
        
        if len(my_samples) > 0:
            # Return the first (and likely only) audio sample
            audio_sample = my_samples[0].detach().squeeze().to("cpu").numpy()
            return (24000, audio_sample), f"✅ Generated audio for: {sanskrit_text}"
        else:
            return None, "❌ Failed to generate audio - no valid codes produced."
            
    except Exception as e:
        return None, f"❌ Error during inference: {str(e)}"

# Load models when the script starts
print("Loading models... This may take a moment.")
load_models()

# Create Gradio interface
with gr.Blocks(title="Sanskrit Text-to-Speech") as demo:
    gr.Markdown("""
    # 🕉️ Sanskrit Text-to-Speech
    
    Enter Sanskrit text in Devanagari script and generate speech using your fine-tuned model.
    Use 1070 in voice if you want most clear output else try 953,643,639,577
    """)
    
    with gr.Row():
        with gr.Column():
            sanskrit_input = gr.Textbox(
                label="Sanskrit Text",
                placeholder="Enter Sanskrit text in Devanagari script...",
                lines=3,
                value="नमस्ते" 
            )
            
            voice_input = gr.Textbox(
                label="Voice (Optional)",
                placeholder="Leave empty for default voice",
                lines=1
            )
            
            generate_btn = gr.Button("🎵 Generate Speech", variant="primary")
        
        with gr.Column():
            audio_output = gr.Audio(
                label="Generated Sanskrit Speech",
                type="numpy"
            )
            
            status_output = gr.Textbox(
                label="Status",
                lines=2,
                interactive=False
            )
    
    # Example inputs
    gr.Examples(
        examples=[
            ["नमस्ते", "1070"],
            ["संस्कृत एक प्राचीन भाषा है।", "1070"],
            ["ॐ शान्ति शान्ति शान्तिः", "1070"],
            ["सर्वे भवन्तु सुखिनः", "1070"],
        ],
        inputs=[sanskrit_input, voice_input],
        outputs=[audio_output, status_output],
        fn=sanskrit_tts_inference,
        cache_examples=False
    )
    
    # Connect the generate button
    generate_btn.click(
        fn=sanskrit_tts_inference,
        inputs=[sanskrit_input, voice_input],
        outputs=[audio_output, status_output]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(
        share=True,  # Set to False if you don't want a public link
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        show_error=True
    )
