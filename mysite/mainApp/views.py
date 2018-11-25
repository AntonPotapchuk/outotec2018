from django.shortcuts import render

# Create your views here.
def index(request):
	return render(request, "mainApp/homePage.html")

def iframe(request):
	return render(request, "mainApp/includes/temp.html")

def contact(request):
	return render(request, "mainApp/basic.html", {"values": ["if you have any question, call me", "+372 56708314"]})