from django import forms

class MatrixForm(forms.Form):
    DECOMP_CHOICES = [
        ('lu', 'Décomposition LU'),
        ('qr', 'Décomposition QR'),
        ('solve', 'Résoudre système linéaire'),
    ]
    
    rows = forms.IntegerField(
        min_value=1,
        max_value=10,
        initial=3,
        label='Nombre de lignes',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'id': 'matrix-rows'})
    )
    
    cols = forms.IntegerField(
        min_value=1,
        max_value=10,
        initial=3,
        label='Nombre de colonnes',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'id': 'matrix-cols'})
    )
    
    matrix = forms.CharField(
        widget=forms.HiddenInput(attrs={'id': 'matrix-data'}),
        label='Matrice'
    )
    
    vector_b = forms.CharField(
        widget=forms.HiddenInput(attrs={'id': 'vector-b-data'}),
        label='Vecteur b',
        required=False
    )
    
    decomposition = forms.ChoiceField(
        choices=DECOMP_CHOICES,
        label='Méthode de décomposition',
        widget=forms.Select(attrs={'class': 'form-select'})
    )