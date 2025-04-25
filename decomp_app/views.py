from django.shortcuts import render
from django.contrib import messages
from .forms import MatrixForm
from .matrix_utils import (
    MatrixError, parse_matrix, lu_decomposition, qr_decomposition, 
    verify_lu, verify_qr, solve_linear_system
)
import json

def index(request):
    """Page d'accueil avec le formulaire"""
    form = MatrixForm()
    return render(request, 'decomp_app/index.html', {'form': form})

def result(request):
    """Page de résultat montrant la décomposition"""
    if request.method != 'POST':
        messages.error(request, "Méthode non autorisée")
        return render(request, 'decomp_app/index.html', {'form': MatrixForm()})
    
    form = MatrixForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Formulaire invalide")
        return render(request, 'decomp_app/index.html', {'form': form})
    
    matrix_str = form.cleaned_data['matrix']
    decomp_method = form.cleaned_data['decomposition']
    
    try:
        matrix = parse_matrix(matrix_str)
        
        # Effectuer la décomposition choisie
        if decomp_method == 'lu':
            L, U = lu_decomposition(matrix)
            original, product = verify_lu(matrix, L, U)
            context = {
                'original': original,
                'decomp_method': 'LU',
                'result': {
                    'L': L,
                    'U': U
                },
                'verification': product,
                'explanation': (
                    "La décomposition LU exprime la matrice comme un produit d'une matrice triangulaire "
                    "inférieure L et d'une matrice triangulaire supérieure U. Elle est utile pour résoudre "
                    "des systèmes d'équations linéaires et calculer des déterminants."
                )
            }
        
        elif decomp_method == 'qr':
            Q, R = qr_decomposition(matrix)
            original, product = verify_qr(matrix, Q, R)
            context = {
                'original': original,
                'decomp_method': 'QR',
                'result': {
                    'Q': Q,
                    'R': R
                },
                'verification': product,
                'explanation': (
                    "La décomposition QR exprime la matrice comme un produit d'une matrice orthogonale Q "
                    "et d'une matrice triangulaire supérieure R. Elle est utilisée pour résoudre des problèmes "
                    "de moindres carrés et calculer des valeurs propres."
                )
            }
        
        elif decomp_method == 'solve':
            vector_b_str = form.cleaned_data['vector_b']
            if not vector_b_str:
                raise MatrixError("Le vecteur b est requis pour résoudre le système linéaire")
            
            b = parse_matrix(vector_b_str)[0]  # Convertir en liste simple
            solution = solve_linear_system(matrix, b)
            
            context = {
                'original': matrix,
                'decomp_method': 'Résolution système linéaire',
                'result': {
                    'solution': solution
                },
                'explanation': (
                    "La solution du système linéaire Ax = b a été trouvée en utilisant la décomposition LU. "
                    "Le vecteur x représente la solution qui satisfait l'équation Ax = b."
                )
            }
        
        else:
            raise MatrixError("Méthode de décomposition inconnue")
        
        # Convertir les matrices pour affichage JSON
        context['original_json'] = json.dumps(context['original'])
        if 'verification' in context:
            context['verification_json'] = json.dumps(context['verification'])
        for key, value in context['result'].items():
            context['result'][key] = json.dumps(value)
        
        return render(request, 'decomp_app/result.html', context)
    
    except MatrixError as e:
        messages.error(request, str(e))
        return render(request, 'decomp_app/index.html', {'form': form})
