import os

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, Spacer
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph

def create_pdf_from_directory(image_dict_path, color_image_path, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    
    # Title style
    title_style = ParagraphStyle(
        name='TitleStyle',
        fontSize=18,
        leading=22,
        textColor='black',  # Customize text color if needed
        fontName='Helvetica-Bold',  # Bold font
        alignment=1,  # Center alignment
    )
    
    # Container for the flowables
    elements = []
    
    for root, dirs, files in os.walk(image_dict_path):
        for file in files:
            image_name = os.path.splitext(file)[0]
            image_path = os.path.join(root, file)
            
            title = Paragraph(f"<b>{image_name}</b>", title_style)
            elements.append(title)
            # Add space between image and previous title or image
            elements.append(Spacer(1, 0.5*inch))
            
            # Add image in the middle
            elements.append(Image(image_path, width=5*inch, height=4*inch))
            
            # Add space between image and color
            elements.append(Spacer(1, 0.25*inch))
            
            color = Paragraph(f"<b>Colors List</b>", title_style)
            elements.append(color)
            
            # Add space between color and color image
            elements.append(Spacer(1, 0.25*inch))
            
            # Add color image at the bottom
            elements.append(Image(color_image_path, width=5*inch, height=0.5*inch))
            
            # Add a page break
            elements.append(PageBreak())
    
    doc.build(elements)

# Example usage
image_dict_path = "output/image"
color_image_path = "output/color_list.png"
output_path = "output/output.pdf"

create_pdf_from_directory(image_dict_path, color_image_path, output_path)
