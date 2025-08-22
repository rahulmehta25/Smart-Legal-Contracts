"""
PDF generation for invoices and receipts.
"""

import io
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from decimal import Decimal

logger = logging.getLogger(__name__)


class PDFInvoiceGenerator:
    """Generate PDF invoices using ReportLab"""
    
    def __init__(self):
        self.company_info = {
            "name": "Arbitration RAG API",
            "address": "123 Tech Street\nSan Francisco, CA 94105",
            "phone": "(555) 123-4567",
            "email": "billing@arbitrationrag.com",
            "website": "www.arbitrationrag.com"
        }
    
    async def generate_invoice_pdf(self, invoice) -> bytes:
        """Generate PDF for invoice"""
        try:
            # Try to import ReportLab
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.lib import colors
                from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
            except ImportError:
                logger.warning("ReportLab not available, generating simple text invoice")
                return await self._generate_text_invoice(invoice)
            
            # Create PDF buffer
            buffer = io.BytesIO()
            
            # Create document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Get styles
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.darkblue,
                alignment=TA_CENTER
            )
            
            header_style = ParagraphStyle(
                'CustomHeader',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.darkblue
            )
            
            # Build story
            story = []
            
            # Company header
            story.append(Paragraph(self.company_info["name"], title_style))
            story.append(Spacer(1, 12))
            
            # Company info
            company_info = f"""
            {self.company_info["address"]}<br/>
            Phone: {self.company_info["phone"]}<br/>
            Email: {self.company_info["email"]}<br/>
            Website: {self.company_info["website"]}
            """
            story.append(Paragraph(company_info, styles['Normal']))
            story.append(Spacer(1, 24))
            
            # Invoice title and number
            story.append(Paragraph("INVOICE", title_style))
            story.append(Spacer(1, 12))
            
            # Invoice details table
            invoice_details = [
                ['Invoice Number:', invoice.invoice_number],
                ['Invoice Date:', invoice.issue_date.strftime("%B %d, %Y")],
                ['Due Date:', invoice.due_date.strftime("%B %d, %Y")],
                ['Status:', invoice.status.title()]
            ]
            
            if invoice.billing_period_start and invoice.billing_period_end:
                invoice_details.append([
                    'Billing Period:',
                    f"{invoice.billing_period_start.strftime('%B %d, %Y')} - {invoice.billing_period_end.strftime('%B %d, %Y')}"
                ])
            
            details_table = Table(invoice_details, colWidths=[2*inch, 3*inch])
            details_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            
            story.append(details_table)
            story.append(Spacer(1, 24))
            
            # Customer information
            customer_data = invoice.customer_data or {}
            story.append(Paragraph("Bill To:", header_style))
            
            customer_info = f"""
            {customer_data.get('full_name', 'N/A')}<br/>
            {customer_data.get('email', 'N/A')}
            """
            
            if customer_data.get('organization'):
                customer_info = f"{customer_data['organization']}<br/>" + customer_info
            
            story.append(Paragraph(customer_info, styles['Normal']))
            story.append(Spacer(1, 24))
            
            # Line items
            story.append(Paragraph("Items:", header_style))
            story.append(Spacer(1, 12))
            
            # Line items table
            line_items_data = [['Description', 'Quantity', 'Unit Price', 'Amount']]
            
            for item in invoice.line_items:
                line_items_data.append([
                    item['description'],
                    str(item['quantity']),
                    f"${item['unit_price']:.2f}",
                    f"${item['amount']:.2f}"
                ])
            
            line_items_table = Table(
                line_items_data,
                colWidths=[3*inch, 1*inch, 1.25*inch, 1.25*inch]
            )
            
            line_items_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),  # Description left-aligned
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(line_items_table)
            story.append(Spacer(1, 24))
            
            # Totals
            totals_data = [
                ['Subtotal:', f"${invoice.subtotal:.2f}"],
                ['Tax:', f"${invoice.tax_amount:.2f}"],
                ['Discount:', f"-${invoice.discount_amount:.2f}"],
                ['Total:', f"${invoice.total_amount:.2f}"]
            ]
            
            totals_table = Table(totals_data, colWidths=[4*inch, 1.5*inch])
            totals_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
                ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LINEBELOW', (0, -1), (-1, -1), 2, colors.black),
            ]))
            
            story.append(totals_table)
            story.append(Spacer(1, 24))
            
            # Payment terms
            story.append(Paragraph("Payment Terms:", header_style))
            payment_terms = f"""
            Payment is due within 30 days of invoice date.<br/>
            Late payments may incur additional fees.<br/>
            Please reference invoice number {invoice.invoice_number} with your payment.
            """
            story.append(Paragraph(payment_terms, styles['Normal']))
            
            # Notes
            if invoice.notes:
                story.append(Spacer(1, 12))
                story.append(Paragraph("Notes:", header_style))
                story.append(Paragraph(invoice.notes, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            # Get PDF content
            pdf_content = buffer.getvalue()
            buffer.close()
            
            return pdf_content
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            # Fallback to text invoice
            return await self._generate_text_invoice(invoice)
    
    async def _generate_text_invoice(self, invoice) -> bytes:
        """Generate simple text invoice as fallback"""
        try:
            customer_data = invoice.customer_data or {}
            
            text_content = f"""
INVOICE

{self.company_info["name"]}
{self.company_info["address"]}
Phone: {self.company_info["phone"]}
Email: {self.company_info["email"]}

=====================================

Invoice Number: {invoice.invoice_number}
Invoice Date: {invoice.issue_date.strftime("%B %d, %Y")}
Due Date: {invoice.due_date.strftime("%B %d, %Y")}
Status: {invoice.status.title()}

=====================================

BILL TO:
{customer_data.get('organization', '')}
{customer_data.get('full_name', 'N/A')}
{customer_data.get('email', 'N/A')}

=====================================

ITEMS:
"""
            
            for item in invoice.line_items:
                text_content += f"""
{item['description']}
Quantity: {item['quantity']}
Unit Price: ${item['unit_price']:.2f}
Amount: ${item['amount']:.2f}
"""
            
            text_content += f"""
=====================================

Subtotal: ${invoice.subtotal:.2f}
Tax: ${invoice.tax_amount:.2f}
Discount: -${invoice.discount_amount:.2f}
TOTAL: ${invoice.total_amount:.2f}

=====================================

PAYMENT TERMS:
Payment is due within 30 days of invoice date.
Please reference invoice number {invoice.invoice_number} with your payment.
"""
            
            if invoice.notes:
                text_content += f"\nNOTES:\n{invoice.notes}\n"
            
            return text_content.encode('utf-8')
            
        except Exception as e:
            logger.error(f"Text invoice generation failed: {e}")
            return b"Invoice generation failed"
    
    async def generate_receipt_pdf(self, payment) -> bytes:
        """Generate PDF receipt for payment"""
        try:
            # Similar to invoice but for payment receipts
            receipt_content = f"""
PAYMENT RECEIPT

{self.company_info["name"]}
{self.company_info["address"]}

Receipt Number: REC-{payment.id}
Payment Date: {payment.processed_at.strftime("%B %d, %Y") if payment.processed_at else 'N/A'}
Payment Method: {payment.payment_method.value.replace('_', ' ').title()}
Amount Paid: ${payment.amount:.2f}
Payment Status: {payment.status.value.title()}

Thank you for your payment!
"""
            
            return receipt_content.encode('utf-8')
            
        except Exception as e:
            logger.error(f"Receipt generation failed: {e}")
            return b"Receipt generation failed"