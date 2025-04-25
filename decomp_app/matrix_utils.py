import math
import copy


class MatrixError(Exception):
    """Exception personnalisée pour les erreurs de matrices"""
    pass

def validate_matrix(matrix):
    """Valide que la matrice est bien formée"""
    if not matrix:
        raise MatrixError("La matrice est vide")
    
    rows = len(matrix)
    if rows == 0:
        raise MatrixError("La matrice doit avoir au moins une ligne")
    
    cols = len(matrix[0])
    if cols == 0:
        raise MatrixError("La matrice doit avoir au moins une colonne")
    
    for row in matrix:
        if len(row) != cols:
            raise MatrixError("Toutes les lignes doivent avoir le même nombre de colonnes")
    
    return rows, cols

def is_square(matrix):
    """Vérifie si la matrice est carrée"""
    rows, cols = validate_matrix(matrix)
    return rows == cols

def transpose(matrix):
    """Calcule la transposée d'une matrice"""
    rows, cols = validate_matrix(matrix)
    return [[matrix[j][i] for j in range(rows)] for i in range(cols)]

def matrix_multiply(A, B):
    """Multiplie deux matrices A et B"""
    rowsA, colsA = validate_matrix(A)
    rowsB, colsB = validate_matrix(B)
    
    if colsA != rowsB:
        raise MatrixError("Dimensions incompatibles pour la multiplication")
    
    C = [[0 for _ in range(colsB)] for _ in range(rowsA)]
    
    for i in range(rowsA):
        for j in range(colsB):
            for k in range(colsA):
                C[i][j] += A[i][k] * B[k][j]
    
    return C

def matrix_vector_multiply(A, v):
    """Multiplie une matrice A par un vecteur v"""
    rowsA, colsA = validate_matrix(A)
    
    if colsA != len(v):
        raise MatrixError("Dimensions incompatibles pour la multiplication matrice-vecteur")
    
    result = [0] * rowsA
    
    for i in range(rowsA):
        for j in range(colsA):
            result[i] += A[i][j] * v[j]
    
    return result

def dot_product(v1, v2):
    """Calcule le produit scalaire de deux vecteurs"""
    if len(v1) != len(v2):
        raise MatrixError("Les vecteurs doivent être de même taille pour le produit scalaire")
    
    return sum(v1[i] * v2[i] for i in range(len(v1)))

def vector_norm(v):
    """Calcule la norme euclidienne d'un vecteur"""
    return math.sqrt(sum(x**2 for x in v))

def identity_matrix(n):
    """Crée une matrice identité de taille n"""
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def zeros_matrix(rows, cols):
    """Crée une matrice de zéros de taille rows x cols"""
    return [[0 for _ in range(cols)] for _ in range(rows)]

# Décomposition LU
def lu_decomposition(matrix):
    """
    Effectue la décomposition LU d'une matrice
    Retourne L, U
    """
    if not is_square(matrix):
        raise MatrixError("La décomposition LU nécessite une matrice carrée")
    
    n = len(matrix)
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # Copie de la matrice pour éviter de la modifier
    A = copy.deepcopy(matrix)
    
    for i in range(n):
        # Diagonale de L est 1
        L[i][i] = 1.0
        
        # Calcul de U
        for j in range(i, n):
            U[i][j] = A[i][j]
            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]
        
        # Calcul de L
        for j in range(i + 1, n):
            L[j][i] = A[j][i]
            for k in range(i):
                L[j][i] -= L[j][k] * U[k][i]
            if abs(U[i][i]) < 1e-10:
                raise MatrixError("La matrice n'est pas décomposable en LU (pivot nul)")
            L[j][i] /= U[i][i]
    
    return L, U

# Décomposition QR via Gram-Schmidt
def qr_decomposition(matrix):
    """
    Effectue la décomposition QR d'une matrice via Gram-Schmidt
    Retourne Q, R
    """
    rows, cols = validate_matrix(matrix)
    
    if cols > rows:
        raise MatrixError("La décomposition QR nécessite une matrice avec plus de lignes que de colonnes")
    
    # Copie de la matrice pour éviter de la modifier
    A = copy.deepcopy(matrix)
    
    Q = []
    R = [[0.0 for _ in range(cols)] for _ in range(cols)]
    
    for j in range(cols):
        # Extraire la j-ème colonne de A
        v = [A[i][j] for i in range(rows)]
        
        # Orthogonalisation de Gram-Schmidt
        for i in range(j):
            # Produit scalaire avec les colonnes précédentes de Q
            R[i][j] = dot_product(Q[i], v)
            # Soustraction de la projection
            for k in range(rows):
                v[k] -= R[i][j] * Q[i][k]
        
        # Normalisation
        norm = vector_norm(v)
        if norm < 1e-10:
            raise MatrixError("Les colonnes de la matrice sont linéairement dépendantes")
        
        R[j][j] = norm
        
        # Ajout de la colonne orthonormée à Q
        q_j = [v[i] / norm for i in range(rows)]
        Q.append(q_j)
    
    # Conversion de Q en matrice (transposée de la liste des colonnes)
    Q_matrix = [[Q[j][i] for j in range(cols)] for i in range(rows)]
    
    return Q_matrix, R

def format_matrix(matrix):
    """Formate une matrice pour l'affichage"""
    return [list(map(lambda x: round(x, 4) if abs(x) > 1e-10 else 0, row)) for row in matrix]

def verify_lu(original, L, U):
    """Vérifie la décomposition LU"""
    LU = matrix_multiply(L, U)
    return format_matrix(original), format_matrix(LU)

def verify_qr(original, Q, R):
    """Vérifie la décomposition QR"""
    QR = matrix_multiply(Q, R)
    return format_matrix(original), format_matrix(QR)

def parse_matrix(matrix_str):
    """Parse une chaîne représentant une matrice"""
    try:
        # Nettoyer la chaîne
        matrix_str = matrix_str.strip()
        
        # Supprimer les crochets extérieurs si présents
        if matrix_str.startswith('[') and matrix_str.endswith(']'):
            matrix_str = matrix_str[1:-1].strip()
        
        # Diviser par lignes
        rows = matrix_str.split(';')
        matrix = []
        
        for row in rows:
            # Nettoyer la ligne
            row = row.strip()
            
            # Supprimer les crochets de ligne si présents
            if row.startswith('[') and row.endswith(']'):
                row = row[1:-1]
            
            # Diviser les éléments et convertir en float
            elements = row.split(',')
            matrix.append([float(e.strip()) for e in elements])
        
        # Vérifier que la matrice est bien formée
        validate_matrix(matrix)
        
        return matrix
    except ValueError:
        raise MatrixError("Format de matrice invalide. Utilisez des nombres séparés par des virgules pour les colonnes et des points-virgules pour les lignes.")
    except MatrixError as e:
        raise e
    except Exception:
        raise MatrixError("Erreur lors de l'analyse de la matrice.")

def solve_lu_system(L, U, b):
    """
    Résout le système linéaire LUx = b en deux étapes:
    1. Résout Ly = b (substitution avant)
    2. Résout Ux = y (substitution arrière)
    """
    n = len(L)
    if n != len(b):
        raise MatrixError("La taille du vecteur b ne correspond pas à la taille de la matrice")
    
    # Résolution de Ly = b (substitution avant)
    y = [0.0 for _ in range(n)]
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]
        y[i] /= L[i][i]
    
    # Résolution de Ux = y (substitution arrière)
    x = [0.0 for _ in range(n)]
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]
    
    return x

def solve_linear_system(matrix, b):
    """
    Résout le système linéaire Ax = b en utilisant la décomposition LU
    """
    if not is_square(matrix):
        raise MatrixError("La matrice doit être carrée pour résoudre un système linéaire")
    
    L, U = lu_decomposition(matrix)
    return solve_lu_system(L, U, b)
