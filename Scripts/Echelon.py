# Andevaldo da Encarnação Vitório

import numpy as np


def show_matrix(matrix, name='ans', decimal_places=2, scientific_notation=True):
  """
  Displays a matrix with the specified number of decimal places.

  Parameters:
  ---
  - matrix: numpy.ndarray, the matrix to be displayed.
  - name: str, the name to be used in the display header (default is 'ans').
  - decimal_places: int, the number of decimal places to display (default is 2).
  - scientific_notation: bool, if True, uses scientific notation; otherwise, uses fixed-point notation (default is True).
  """
  # Define the format pattern based on whether scientific notation is used
  pattern = f"{{:.{decimal_places}{'e' if scientific_notation else 'f'}}}"

  def format_elem(elem):
    """
    Formats a single matrix element according to the defined pattern.

    Parameters:
    - elem: the element to be formatted.

    Returns:
    - str: the formatted element.
    """
    return pattern.format(elem)

  # Calculate the maximum width required for each column
  col_widths = [max(map(len, map(format_elem, col))) for col in matrix.T]

  print(f"{name} =")  # Print the matrix name
  # Calculate the spacing for the matrix border
  nspaces = sum(col_widths) + 2 * matrix.shape[1]

  # Print the top border of the matrix
  print("    ┌" + " " * nspaces + "┐")

  # Print each row of the matrix
  for row in matrix:
    # Format each element of the row and right-align according to column width
    formatted_row = "  ".join(format_elem(e).rjust(w)
                              for e, w in zip(row, col_widths))
    print(f"    │ {formatted_row} │")

  # Print the bottom border of the matrix
  print("    └" + " " * nspaces + "┘\n")

def row_echelon_form(A, reduced=False):
  """
  Reduces the matrix A to row echelon form or reduced row echelon form.

  Parameters:
  A: input matrix.
  reduced: boolean, if True, returns the reduced row echelon form (RREF).
           If False, returns only the row echelon form (REF).

  Returns:
  A: matrix reduced to the desired form.
  """
  A = np.array(
      A, dtype=float)  # Convert the matrix to float to ensure precision
  rows, cols = A.shape

  # Reduce the matrix to row echelon form (REF)
  for col in range(min(rows, cols)):
    # Choose the pivot
    max_row = np.argmax(np.abs(A[col:rows, col])) + col
    if A[max_row, col] == 0:
      continue  # Skip the column if the pivot is zero

    # Swap the current row with the row containing the pivot
    A[[col, max_row]] = A[[max_row, col]]

    # Normalize the pivot row so that the pivot is 1
    A[col] = A[col] / A[col, col]

    # Eliminate the values below the pivot
    for i in range(col + 1, rows):
      A[i] -= A[i, col] * A[col]

  # If reduced is True, reduce to the reduced row echelon form (RREF)
  if reduced:
    for col in range(min(rows, cols) - 1, -1, -1):
      pivot_row = None
      for row in range(rows):
        if A[row, col] == 1:
          pivot_row = row
          break

      if pivot_row is not None:
        for row in range(pivot_row):
          A[row] -= A[row, col] * A[pivot_row]

  return A


A = [[1, 3, 3, 8, 5],
     [0, 1, 3, 10, 8],
     [0, 0, 0, -1, -4],
     [0, 0, 0, 2, 8]]

# To obtain the row echelon form (REF)
echelon_form = row_echelon_form(A, reduced=False)
print("Row Echelon Form:")
show_matrix(echelon_form, scientific_notation=False)

# To obtain the reduced row echelon form (RREF)
reduced_echelon_form = row_echelon_form(A, reduced=True)
print("\nReduced Row Echelon Form:")
show_matrix(reduced_echelon_form, scientific_notation=False)
