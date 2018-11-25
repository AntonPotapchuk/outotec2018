from django.db import models

# Create your models here.
class Articles(models.Model):
	"""docstring for Articles"""
	title = models.CharField(max_length = 120)
	post = models.TextField()
	date = models.DateTimeField()

	def __str__(self):     #when we call Articles title it will show title and not something unexpected
		return self.title
		