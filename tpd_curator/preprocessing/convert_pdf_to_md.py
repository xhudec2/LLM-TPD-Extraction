from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
import os


def convert_pdf_to_md(config_parser, paper_path, output_dir, save_images=False):
    print("Start converting PDF to markdown...")

    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )

    if os.path.isfile(paper_path):
        rendered = converter(paper_path)
        text, _, images = text_from_rendered(rendered)
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(paper_path))[0]
        out_file_name = base_name + ".md"
        text_file = os.path.join(output_dir, out_file_name)
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Text saved to: {text_file}")

        if save_images:
            for image_name, img in images.items():
                img.save(os.path.join(output_dir, image_name), "JPEG")

    elif os.path.isdir(paper_path):
        for root, _, files in os.walk(paper_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    rendered = converter(file_path)
                    text, _, images = text_from_rendered(rendered)
                    base_name = os.path.splitext(file)[0]
                    pdf_output_dir = os.path.join(output_dir, base_name)
                    os.makedirs(pdf_output_dir, exist_ok=True)
                    text_file = os.path.join(pdf_output_dir, base_name + ".md")
                    with open(text_file, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"Converted and saved: {text_file}")

                    if save_images:
                        for image_name, img in images.items():
                            img.save(os.path.join(pdf_output_dir, image_name), "JPEG")
    else:
        print(
            "The input path is invalid, please check if it is the correct file or folder path."
        )
