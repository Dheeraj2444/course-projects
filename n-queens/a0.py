#!/usr/bin/env python
# Solve the N-Rooks and N-Queens problem!
# D. Crandall, 2016
# Updated by Zehua Zhang, 2017

'''
Assignment 0
Optimised code for N-Queens and N-Rooks
Submitted By: Dheeraj Singh
'''

import sys

# Count # of pieces in given row
def count_on_row(board, row):
    return sum( board[row] ) 

# Count # of pieces in given column
def count_on_col(board, col):
    return sum( [ row[col] for row in board ] ) 

# Count # of pieces in diagonal
def count_on_diagonal(board, row, col):
    diagonal_elements = [board[row][col]]
    row1 = row
    col1 = col
    while row1<N-1 and col1<N-1:
        row1, col1=row1+1, col1+1
        diagonal_elements.append(board[row1][col1])
    row1 = row
    col1 = col
    while row1>0 and col1>0:
        row1, col1 = row1-1, col1-1
        diagonal_elements.append(board[row1][col1])
    row1 = row
    col1 = col
    while col1<N-1 and row1>0:
        row1, col1 = row1-1, col1+1
        diagonal_elements.append(board[row1][col1])
    row1 = row
    col1 = col
    while col1>0 and row1 < N-1:
        row1, col1 = row1+1, col1-1
        diagonal_elements.append(board[row1][col1])

    return sum(diagonal_elements)

# Count total # of pieces on board
def count_pieces(board):
    return sum([ sum(row) for row in board ] )

# Return a string with the board rendered in a human-friendly format
def printable_board(board):
    if method=="nrook":
        return "\n".join([ " ".join([ "R" if col==1 else "X" if col=="X" else "_" for col in row ]) for row in board])
    else:
        return "\n".join([ " ".join([ "Q" if col==1 else "X" if col=="X" else "_" for col in row ]) for row in board])

# Add a piece to the board at the given position, and return a new board (doesn't change original)
def add_piece(board, row, col):
    return board[0:row] + [board[row][0:col] + [1,] + board[row][col+1:]] + board[row+1:]

# Get list of successors of given board state
def successors(board):
    left_most_empty_col = 0   #implemented this idea of adding a piece to the left-most empty column from the book: Page 72
    for col in range(0, N):
        if count_on_col(board, col)>0:
            left_most_empty_col+=1
    if method=="nrook":
        return [ add_piece(board, r, left_most_empty_col) for r in range(0, N) if count_on_row(board,r)==0 and [r,left_most_empty_col]!=[invalid_row,invalid_col]]
    else:
        return [ add_piece(board, r, left_most_empty_col) for r in range(0, N) if count_on_row(board,r)==0 and count_on_diagonal(board,r,left_most_empty_col)==0 and [r,left_most_empty_col]!=[invalid_row,invalid_col]]
    
# check if board is a goal state
def is_goal(board):
    if method=="nrook":
        return count_pieces(board) == N and \
            all( [ count_on_row(board, r) <= 1 for r in range(0, N) ] ) and \
            all( [ count_on_col(board, c) <= 1 for c in range(0, N) ] )
    else:
        return count_pieces(board) == N and \
            all( [ count_on_row(board, r) <= 1 for r in range(0, N) ] ) and \
            all( [ count_on_col(board, c) <= 1 for c in range(0, N) ] ) and \
            all( [ count_on_diagonal(board, r, c) <= 1 for r in range(0, N) for c in range(0, N) if board[r][c]==1 ] )
        
# Solve n-rooks!
def solve(initial_board):
    fringe = [initial_board]
    visited = []             #implemented this idea of storing visited nodes from the book: Page 77
    while len(fringe) > 0:
        for s in successors( fringe.pop() ):
            if s not in visited:
                if is_goal(s):
                    return(s)
                visited.append(s)
                fringe.append(s)
    return False

method = (sys.argv[1])                            #user input for nqueen or nrook
N = int(sys.argv[2])                              #user input for value of N
invalid_row = None
invalid_col = None                                #when third and fourth argument are 0; all positions are available
if int(sys.argv[3])!=0:                           
    invalid_row = int(sys.argv[3])-1              #user input for invalid row number
if int(sys.argv[4])!=0:
    invalid_col = int(sys.argv[4])-1              #user input for invalid column number            

def main():     
    if len(sys.argv)!=5 or method not in ['nqueen', 'nrook'] or (invalid_row!=None and invalid_row+1>N) or (invalid_col!=None and invalid_col+1>N):
        print "Invalid input; arguments must be in the format: 'nrook 7 1 1' or 'nqueen 7 1 1'; last two arguments cannot be greater than second"
    else:
        initial_board = [[0]*N]*N
        solution = solve(initial_board)
        if solution:
            if invalid_row==None or invalid_col==None:
                print printable_board(solution)
            else:
                solution[invalid_row][invalid_col]="X"
                print printable_board(solution) 
        else:
            print "Sorry, no solution found. :("

if __name__ == "__main__":
    main()
