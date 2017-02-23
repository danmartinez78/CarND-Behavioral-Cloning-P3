from markdown import markdown
import pdfkit

input_filename = 'writeup_template.md'
output_filename = 'writeup_report.pdf'

with open(input_filename, 'r') as f:
    html_text = markdown(f.read(), output_format='html4')

pdfkit.from_string(html_text, output_filename)