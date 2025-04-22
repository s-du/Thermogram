"""
Report generation tools for DroneIR Toolkit.
- Automated Word report creation with sections:
  Contents / Objectives / Site and Conditions / Flight details / Images
- Images section is filled with selected images (ProcessedIm)
- Embedded annotations summarized as tables
"""
import os
import io
import tempfile
import numpy as np
from PIL import Image
import cv2
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.table import _Cell
from .core import ProcessedIm, cv_read_all_path

STYLE_TEMPLATES = {
    "modern_blue": {
        "Heading 1": {"font": "Arial", "size": 18, "bold": True, "color": (0x2E, 0x74, 0xB5)},
        "Heading 2": {"font": "Arial", "size": 15, "bold": True, "color": (0x4F, 0x81, 0xBD)},
        "Heading 3": {"font": "Arial", "size": 13, "bold": True, "color": (0x1F, 0x49, 0x7D)},
        "Normal":    {"font": "Calibri", "size": 11, "color": (0, 0, 0)},
        "Table Grid": {"font": "Calibri", "size": 10, "color": (0, 0, 0), "shading": (0xF2, 0xF9, 0xFF)}
    },
    "classic_gray": {
        "Heading 1": {"font": "Times New Roman", "size": 20, "bold": True, "color": (0x44, 0x44, 0x44)},
        "Heading 2": {"font": "Times New Roman", "size": 16, "bold": True, "color": (0x66, 0x66, 0x66)},
        "Heading 3": {"font": "Times New Roman", "size": 13, "bold": True, "color": (0x88, 0x88, 0x88)},
        "Normal":    {"font": "Times New Roman", "size": 12, "color": (0x33, 0x33, 0x33)},
        "Table Grid": {"font": "Times New Roman", "size": 10, "color": (0x44, 0x44, 0x44), "shading": (0xF6, 0xF6, 0xF6)}
    },
    "dark_elegant": {
        "Heading 1": {"font": "Segoe UI", "size": 18, "bold": True, "color": (0xFF, 0xFF, 0xFF), "bg": (0x22, 0x22, 0x22)},
        "Heading 2": {"font": "Segoe UI", "size": 15, "bold": True, "color": (0xEE, 0xEE, 0xEE), "bg": (0x33, 0x33, 0x33)},
        "Heading 3": {"font": "Segoe UI", "size": 13, "bold": True, "color": (0xCC, 0xCC, 0xCC), "bg": (0x44, 0x44, 0x44)},
        "Normal":    {"font": "Segoe UI", "size": 11, "color": (0x22, 0x22, 0x22)},
        "Table Grid": {"font": "Segoe UI", "size": 10, "color": (0x22, 0x22, 0x22), "shading": (0xE5, 0xE5, 0xE5)}
    }
}

def set_custom_styles(doc, style_template="modern_blue"):
    """
    Apply the chosen style template to the Word document.
    """
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls
    styles = doc.styles
    tpl = STYLE_TEMPLATES.get(style_template, STYLE_TEMPLATES["modern_blue"])
    for style_name in ["Heading 1", "Heading 2", "Heading 3", "Normal", "Table Grid"]:
        if style_name not in styles:
            continue
        style = styles[style_name]
        sdef = tpl.get(style_name, {})
        if hasattr(style, "font"):
            if "font" in sdef: style.font.name = sdef["font"]
            if "size" in sdef: style.font.size = Pt(sdef["size"])
            if "bold" in sdef: style.font.bold = sdef["bold"]
            if "color" in sdef: style.font.color.rgb = RGBColor(*sdef["color"])
        # Custom background for headings (optional, only for dark_elegant)
        if style_name.startswith("Heading") and "bg" in sdef:
            for p in doc.paragraphs:
                if p.style.name == style_name:
                    p._element.get_or_add_pPr().insert(0, parse_xml(
                        r'<w:shd {} w:fill="{:02X}{:02X}{:02X}"/>'.format(
                            nsdecls('w'), *sdef["bg"])))
    # Table style shading (applied later per-table)
    return tpl


def add_table_from_annotations(doc, annotations, title="Annotations Summary", style_template=None, annotation_type=None):
    """
    Add a table summarizing annotations to the doc.
    annotations: list of dicts or objects with annotation info
    style_template: dict with table style settings (optional)
    annotation_type: type of annotation ('rect', 'point', 'line') to customize display
    """
    if not annotations:
        return
    
    doc.add_heading(title, level=3)
    
    # Convert objects to dicts if needed
    ann_dicts = []
    for ann in annotations:
        if not isinstance(ann, dict):
            ann_dict = {k: v for k, v in ann.__dict__.items() 
                      if not k.startswith('_') and not callable(v) and not isinstance(v, (list, tuple, dict)) 
                      and not any(excluded in k for excluded in ['item', 'ellipse', 'text', 'spot', 'main'])}
            ann_dicts.append(ann_dict)
        else:
            ann_dicts.append(ann)
    
    # Select relevant keys based on annotation type
    if annotation_type == 'rect':
        # For rectangular measurements, focus on temperature data and area
        keys = ['name', 'tmin', 'tmax', 'tmean', 'tstd', 'area']
        headers = ['Name', 'Min Temp (°C)', 'Max Temp (°C)', 'Mean Temp (°C)', 'Std Dev (°C)', 'Area (px²)']
    elif annotation_type == 'point':
        # For point measurements, show position and temperature
        keys = ['name', 'temp']
        headers = ['Name', 'Temperature (°C)']
    elif annotation_type == 'line':
        # For line measurements, show min/max/mean temperatures
        keys = ['name', 'tmin', 'tmax', 'tmean']
        headers = ['Name', 'Min Temp (°C)', 'Max Temp (°C)', 'Mean Temp (°C)']
    else:
        # Default: use all available keys (filtered)
        keys = []
        for ann in ann_dicts:
            for k in ann.keys():
                if k not in keys and not k.startswith('_') and not any(excluded in k for excluded in ['item', 'ellipse', 'text', 'spot', 'main', 'coords', 'data_roi']):
                    keys.append(k)
        headers = [k.replace('_', ' ').title() for k in keys]
    
    # Create table with selected keys
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'
    
    # Apply table shading if specified
    if style_template and "Table Grid" in style_template:
        shading = style_template["Table Grid"].get("shading")
        if shading:
            from docx.oxml import parse_xml
            from docx.oxml.ns import nsdecls
            for row in table.rows:
                for cell in row.cells:
                    cell._tc.get_or_add_tcPr().append(parse_xml(
                        r'<w:shd {} w:fill="{:02X}{:02X}{:02X}"/>'.format(
                            nsdecls('w'), *shading)))
    
    # Set headers
    hdr_cells = table.rows[0].cells
    for i, header in enumerate(headers):
        hdr_cells[i].text = header
        # Make headers bold
        for paragraph in hdr_cells[i].paragraphs:
            for run in paragraph.runs:
                run.bold = True
    
    # Fill table with data
    for ann in ann_dicts:
        row_cells = table.add_row().cells
        for i, key in enumerate(keys):
            value = ann.get(key, 'N/A')
            # Format numeric values
            if isinstance(value, float):
                row_cells[i].text = f"{value:.2f}"
            else:
                row_cells[i].text = str(value)
    
    doc.add_paragraph()

def safe_add_image_to_doc(doc, image_path, width=None, height=None):
    """
    Safely add an image to a Word document, handling various image formats
    and potential issues with thermal images.
    
    Args:
        doc: The Word document object
        image_path: Path to the image file
        width: Optional width for the image
        height: Optional height for the image
    """
    try:
        # First try direct addition (fastest)
        doc.add_picture(image_path, width=width, height=height)
    except Exception:
        # If direct addition fails, try conversion with PIL
        try:
            # Open with PIL which handles more formats
            with Image.open(image_path) as img:
                # Convert to RGB if needed (some thermal images might be in unusual formats)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_path = temp_file.name
                    img.save(temp_path, format='PNG')
                
                # Add the converted image
                doc.add_picture(temp_path, width=width, height=height)
                
                # Clean up
                os.unlink(temp_path)
        except Exception:
            # If PIL fails, try with OpenCV
            try:
                # Read with OpenCV
                img = cv_read_all_path(image_path)
                if img is None:
                    raise ValueError("Could not read image with OpenCV")
                    
                # Convert BGR to RGB if needed
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_path = temp_file.name
                    cv2.imwrite(temp_path, img)
                
                # Add the converted image
                doc.add_picture(temp_path, width=width, height=height)
                
                # Clean up
                os.unlink(temp_path)
            except Exception as e:
                # If all methods fail, re-raise the exception
                raise ValueError(f"Could not process image: {str(e)}")


def create_word_report(
    output_path,
    objectives_text,
    site_conditions_text,
    flight_details_text,
    processed_images,
    images_to_include=None,
    style_template="modern_blue",
    include_summary=True
):
    """
    Create a Word report with specified sections and images.
    processed_images: list of ProcessedIm objects
    images_to_include: list of indices or ProcessedIm objects to include (default: all)
    style_template: string key for the style template to use
    """
    doc = Document()
    tpl = set_custom_styles(doc, style_template)
    doc.add_heading('Infrared Survey Report', 0)

    # Contents
    doc.add_heading('Contents', level=1)
    toc = doc.add_paragraph()
    toc.add_run('1. Objectives\n').bold = True
    toc.add_run('2. Site and Conditions\n').bold = True
    toc.add_run('3. Flight details\n').bold = True
    toc.add_run('4. Images\n').bold = True
    doc.add_page_break()

    # Objectives
    doc.add_heading('Objectives', level=1)
    doc.add_paragraph(objectives_text)
    doc.add_page_break()

    # Site and Conditions
    doc.add_heading('Site and Conditions', level=1)
    doc.add_paragraph(site_conditions_text)
    doc.add_page_break()

    # Flight details
    doc.add_heading('Flight details', level=1)
    doc.add_paragraph(flight_details_text)
    doc.add_page_break()

    # Images Section
    doc.add_heading('Images', level=1)
    
    # Add summary section if requested
    if include_summary and processed_images:
        doc.add_heading('Summary of Findings', level=2)
        summary_para = doc.add_paragraph()
        summary_para.add_run('This section provides a summary of key thermal findings across all analyzed images.')
        
        # Create summary table of key measurements
        all_rect_measurements = []
        for img_idx, img in enumerate(processed_images):
            pim = img if isinstance(img, ProcessedIm) else img
            if hasattr(pim, 'meas_rect_list') and pim.meas_rect_list:
                for rect in pim.meas_rect_list:
                    if hasattr(rect, 'tmax') and hasattr(rect, 'tmin'):
                        name = getattr(rect, 'name', f'Area in Image {img_idx}')
                        all_rect_measurements.append({
                            'Image': img_idx,
                            'Name': name,
                            'Min Temp': getattr(rect, 'tmin', 'N/A'),
                            'Max Temp': getattr(rect, 'tmax', 'N/A'),
                            'Delta T': getattr(rect, 'tmax', 0) - getattr(rect, 'tmin', 0) if hasattr(rect, 'tmax') and hasattr(rect, 'tmin') else 'N/A'
                        })
        
        if all_rect_measurements:
            # Sort by max temperature to highlight potential issues
            all_rect_measurements.sort(key=lambda x: x['Max Temp'] if isinstance(x['Max Temp'], (int, float)) else -999, reverse=True)
            
            # Add summary table
            doc.add_heading('Temperature Hotspots', level=3)
            table = doc.add_table(rows=1, cols=5)
            table.style = 'Table Grid'
            
            # Headers
            headers = ['Image', 'Name', 'Min Temp (°C)', 'Max Temp (°C)', 'Delta T (°C)'] 
            hdr_cells = table.rows[0].cells
            for i, header in enumerate(headers):
                hdr_cells[i].text = header
                for paragraph in hdr_cells[i].paragraphs:
                    for run in paragraph.runs:
                        run.bold = True
            
            # Data rows (limit to top 5 for clarity)
            for measurement in all_rect_measurements[:5]:
                row_cells = table.add_row().cells
                row_cells[0].text = str(measurement['Image'])
                row_cells[1].text = str(measurement['Name'])
                row_cells[2].text = f"{measurement['Min Temp']:.2f}" if isinstance(measurement['Min Temp'], (int, float)) else str(measurement['Min Temp'])
                row_cells[3].text = f"{measurement['Max Temp']:.2f}" if isinstance(measurement['Max Temp'], (int, float)) else str(measurement['Max Temp'])
                row_cells[4].text = f"{measurement['Delta T']:.2f}" if isinstance(measurement['Delta T'], (int, float)) else str(measurement['Delta T'])
            
            doc.add_paragraph()
        
        doc.add_page_break()
    
    # Individual images
    if images_to_include is None:
        images_to_include = list(range(len(processed_images)))
    for idx in images_to_include:
        pim = processed_images[idx] if isinstance(processed_images[idx], ProcessedIm) else processed_images[idx]
        doc.add_heading(f'Image {idx}', level=2)
        # Add thermal image if available
        if pim.path and os.path.exists(pim.path):
            doc.add_heading('Thermal Image', level=3)
            try:
                # Try to safely add the thermal image using PIL to convert if needed
                safe_add_image_to_doc(doc, pim.path, width=Inches(4))
            except Exception as e:
                # If thermal image fails, add a note
                doc.add_paragraph(f"[Thermal image could not be displayed: {str(e)}]", style='Normal')
            
        # Add RGB image if available
        if pim.rgb_path and os.path.exists(pim.rgb_path):
            doc.add_heading('RGB Image', level=3)
            try:
                # Try to safely add the RGB image
                safe_add_image_to_doc(doc, pim.rgb_path, width=Inches(4))
            except Exception as e:
                # If RGB image fails, add a note
                doc.add_paragraph(f"[RGB image could not be displayed: {str(e)}]", style='Normal')
        # Add image info
        para1 = doc.add_paragraph()
        run1 = para1.add_run(f'Original: {os.path.basename(pim.rgb_path_original)}')
        run1.font.size = Pt(10)
        run1.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
        para2 = doc.add_paragraph()
        run2 = para2.add_run(f'Drone model: {getattr(pim, "drone_model", "N/A")}')
        run2.font.size = Pt(10)
        run2.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
        # Add key thermal information section
        doc.add_heading('Thermal Analysis', level=2)
        
        # Add temperature range information
        if hasattr(pim, 'tmin') and hasattr(pim, 'tmax'):
            temp_para = doc.add_paragraph()
            temp_para.add_run('Temperature Range: ').bold = True
            temp_para.add_run(f"{pim.tmin:.2f}°C to {pim.tmax:.2f}°C")
        
        # Add annotation tables if available with improved formatting
        if hasattr(pim, 'annot_rect_items') and pim.annot_rect_items:
            add_table_from_annotations(doc, pim.annot_rect_items, title="Rectangular Annotations", 
                                      style_template=tpl, annotation_type='rect')
        
        if hasattr(pim, 'meas_rect_list') and pim.meas_rect_list:
            add_table_from_annotations(doc, pim.meas_rect_list, title="Area Measurements", 
                                      style_template=tpl, annotation_type='rect')
        
        if hasattr(pim, 'meas_point_list') and pim.meas_point_list:
            add_table_from_annotations(doc, pim.meas_point_list, title="Spot Measurements", 
                                      style_template=tpl, annotation_type='point')
        
        if hasattr(pim, 'meas_line_list') and pim.meas_line_list:
            add_table_from_annotations(doc, pim.meas_line_list, title="Line Profile Measurements", 
                                      style_template=tpl, annotation_type='line')
        doc.add_page_break()

    doc.save(output_path)
    return output_path
