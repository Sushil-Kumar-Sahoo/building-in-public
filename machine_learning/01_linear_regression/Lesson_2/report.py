from reportlab.pdfgen import canvas

c = canvas.Canvas("boston_housing_analysis.pdf")
c.draw(220,810,"<h1>Boston Housing</h1>")
c.save()