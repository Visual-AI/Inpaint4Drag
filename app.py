import gradio as gr
from utils.ui_utils import *

CANVAS_SIZE = 400
DEFAULT_GEN_SIZE = 512

def create_interface():
    with gr.Blocks() as app:
        # State variables
        state = {
            'canvas_size': gr.Number(value=CANVAS_SIZE, visible=False, precision=0),
            'gen_size': gr.Number(value=DEFAULT_GEN_SIZE, visible=False, precision=0),
            'points_list': gr.State(value=[]),
            'inpaint_mask': gr.State(value=None)
        }
        
        with gr.Tab(label='Inpaint4Drag'):
            with gr.Row():
                # Draw Region Column
                with gr.Column():
                    gr.Markdown("""<p style="text-align: center; font-size: 20px">1. Draw Regions</p>""")
                    canvas = gr.Image(type="numpy", tool="sketch", label=" ", height=CANVAS_SIZE, width=CANVAS_SIZE)
                    with gr.Row():
                        fit_btn = gr.Button("Resize Image")
                        if_sam_box = gr.Checkbox(label='Refine mask (SAM)')

                # Control Points Column
                with gr.Column():
                    gr.Markdown("""<p style="text-align: center; font-size: 20px">2. Control Points</p>""")
                    input_img = gr.Image(type="numpy", label=" ", height=CANVAS_SIZE, width=CANVAS_SIZE, interactive=True)
                    with gr.Row():
                        undo_btn = gr.Button("Undo Point")
                        clear_btn = gr.Button("Clear Points")

                # Results Column
                with gr.Column():
                    gr.Markdown("""<p style="text-align: center; font-size: 20px">Results</p>""")
                    output_img = gr.Image(type="numpy", label=" ", height=CANVAS_SIZE, width=CANVAS_SIZE, interactive=False)
                    with gr.Row():
                        run_btn = gr.Button("Inpaint")
                        reset_btn = gr.Button("Reset All")

        # Output Settings
        with gr.Row("Generation Parameters"):
            sam_ks = gr.Slider(minimum=11, maximum=51, value=21, step=2, label='How much to refine mask with SAM', interactive=True)
            inpaint_ks = gr.Slider(minimum=0, maximum=25, value=5, step=1, label='How much to expand inpainting mask', interactive=True)
            output_path = gr.Textbox(value='output/app', label="Output path")

        setup_events(
            components={
                'canvas': canvas,
                'input_img': input_img,
                'output_img': output_img,
                'output_path': output_path,
                'if_sam_box': if_sam_box,
                'sam_ks': sam_ks,
                'inpaint_ks': inpaint_ks,
            },
            state=state,
            buttons={
                'fit': fit_btn,
                'undo': undo_btn,
                'clear': clear_btn,
                'run': run_btn,
                'reset': reset_btn
            }
        )

    return app

def setup_events(components, state, buttons):
    # Reset and clear events
    def setup_reset_events():
        buttons['reset'].click(
            clear_all,
            [state['canvas_size']],
            [components['canvas'], components['input_img'], components['output_img'], 
             state['points_list'], components['sam_ks'], components['inpaint_ks'], components['output_path'], state['inpaint_mask']]
        )
        
        components['canvas'].clear(
            clear_all,
            [state['canvas_size']],
            [components['canvas'], components['input_img'], components['output_img'], 
             state['points_list'], components['sam_ks'], components['inpaint_ks'], components['output_path'], state['inpaint_mask']]
        )

    # Image manipulation events
    def setup_image_events():
        buttons['fit'].click(
            clear_point,
            [components['canvas'], state['points_list'], components['sam_ks'], components['if_sam_box'], components['output_path']],
            [components['input_img']]
        ).then(
            resize,
            [components['canvas'], state['gen_size'], state['canvas_size']],
            [components['canvas'], components['input_img'], components['output_img']]
        )

    # Canvas interaction events
    def setup_canvas_events():
        components['canvas'].edit(
            visualize_user_drag,
            [components['canvas'], state['points_list'], components['sam_ks'], components['if_sam_box'], components['output_path']],
            [components['input_img']]
        ).then(
            preview_out_image,
            [components['canvas'], state['points_list'], components['sam_ks'], components['inpaint_ks'], components['if_sam_box'], components['output_path']],
            [components['output_img'], state['inpaint_mask']]
        )

        components['if_sam_box'].change(
            visualize_user_drag,
            [components['canvas'], state['points_list'], components['sam_ks'], components['if_sam_box']],
            [components['input_img']]
        ).then(
            preview_out_image,
            [components['canvas'], state['points_list'], components['sam_ks'], components['inpaint_ks'], components['if_sam_box'], components['output_path']],
            [components['output_img'], state['inpaint_mask']]
        )

        components['sam_ks'].change(
            visualize_user_drag,
            [components['canvas'], state['points_list'], components['sam_ks'], components['if_sam_box']],
            [components['input_img']]
        ).then(
            preview_out_image,
            [components['canvas'], state['points_list'], components['sam_ks'], components['inpaint_ks'], components['if_sam_box'], components['output_path']],
            [components['output_img'], state['inpaint_mask']]
        )
        
        components['inpaint_ks'].change(
            visualize_user_drag,
            [components['canvas'], state['points_list'], components['sam_ks'], components['if_sam_box']],
            [components['input_img']]
        ).then(
            preview_out_image,
            [components['canvas'], state['points_list'], components['sam_ks'], components['inpaint_ks'], components['if_sam_box'], components['output_path']],
            [components['output_img'], state['inpaint_mask']]
        )

    # Input image events
    def setup_input_events():
        components['input_img'].select(
            add_point,
            [components['canvas'], state['points_list'], components['sam_ks'], components['if_sam_box'], components['output_path']],
            [components['input_img']]
        ).then(
            preview_out_image,
            [components['canvas'], state['points_list'], components['sam_ks'], components['inpaint_ks'], components['if_sam_box'], components['output_path']],
            [components['output_img'], state['inpaint_mask']]
        )

    # Point manipulation events
    def setup_point_events():
        buttons['undo'].click(
            undo_point,
            [components['canvas'], state['points_list'], components['sam_ks'], components['if_sam_box'], components['output_path']],
            [components['input_img']]
        ).then(
            preview_out_image,
            [components['canvas'], state['points_list'], components['sam_ks'], components['inpaint_ks'], components['if_sam_box'], components['output_path']],
            [components['output_img'], state['inpaint_mask']]
        )
        
        buttons['clear'].click(
            clear_point,
            [components['canvas'], state['points_list'], components['sam_ks'], components['if_sam_box'], components['output_path']],
            [components['input_img']]
        ).then(
            preview_out_image,
            [components['canvas'], state['points_list'], components['sam_ks'], components['inpaint_ks'], components['if_sam_box'], components['output_path']],
            [components['output_img'], state['inpaint_mask']]
        )

    # Processing events
    def setup_processing_events():
        buttons['run'].click(
            preview_out_image,
           [components['canvas'], state['points_list'], components['sam_ks'], components['inpaint_ks'], components['if_sam_box'], components['output_path']],
            [components['output_img'], state['inpaint_mask']]
        ).then(
            inpaint,
            [components['output_img'], state['inpaint_mask']],
            [components['output_img']]
        )

    # Setup all events
    setup_reset_events()
    setup_image_events()
    setup_canvas_events()
    setup_input_events()
    setup_point_events()
    setup_processing_events()

def main():
    app = create_interface()
    app.queue().launch(share=True, debug=True)

if __name__ == '__main__':
    main()