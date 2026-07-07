"""
Export utilities for the inventory simulator.
Supports CSV and Excel export with formatting.
"""

import pandas as pd
import io
from typing import Optional


def export_to_csv(
    df: pd.DataFrame,
    filename: str = "export.csv",
    index: bool = False,
    encoding: str = "utf-8-sig"
) -> str:
    """
    Convert DataFrame to CSV format for download.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to export
    filename : str
        Suggested filename
    index : bool
        Whether to include index in export
    encoding : str
        File encoding (default: utf-8-sig for Excel compatibility)
    
    Returns:
    --------
    str
        CSV content as string
    """
    return df.to_csv(index=index, encoding=encoding)


def export_to_excel(
    df: pd.DataFrame,
    filename: str = "export.xlsx",
    sheet_name: str = "Datos",
    include_formatting: bool = True
) -> bytes:
    """
    Convert DataFrame to Excel format with optional formatting.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to export
    filename : str
        Suggested filename (not used in output, just for reference)
    sheet_name : str
        Name of the Excel sheet
    include_formatting : bool
        Whether to apply formatting (headers, column widths, etc.)
    
    Returns:
    --------
    bytes
        Excel file content as bytes
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        if include_formatting:
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
            # Format headers
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)
                cell.fill = cell.fill.copy(
                    start_color='366092',
                    end_color='366092'
                )
                cell.font = cell.font.copy(color='FFFFFF')
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    return output.getvalue()


def export_multiple_sheets(
    data_dict: dict,
    filename: str = "report.xlsx"
) -> bytes:
    """
    Export multiple DataFrames to an Excel file with multiple sheets.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary where keys are sheet names and values are DataFrames
    filename : str
        Suggested filename (for reference)
    
    Returns:
    --------
    bytes
        Excel file content as bytes
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in data_dict.items():
            # Clean sheet name (max 31 chars, no special chars)
            clean_name = sheet_name[:31].replace('/', '-').replace('\\', '-')
            df.to_excel(writer, sheet_name=clean_name, index=False)
            
            # Apply basic formatting
            worksheet = writer.sheets[clean_name]
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)
    
    return output.getvalue()


def create_download_button_data(
    df: pd.DataFrame,
    format: str = "csv",
    **kwargs
) -> tuple:
    """
    Create data for a Streamlit download button.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to export
    format : str
        Export format: 'csv' or 'xlsx'
    **kwargs
        Additional arguments passed to export functions
    
    Returns:
    --------
    tuple
        (filename, file_content, mime_type)
    """
    if format.lower() == "csv":
        filename = kwargs.get('filename', 'data.csv')
        content = export_to_csv(df, filename, **kwargs)
        mime_type = "text/csv"
    elif format.lower() == "xlsx":
        filename = kwargs.get('filename', 'data.xlsx')
        content = export_to_excel(df, filename, **kwargs)
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return filename, content, mime_type