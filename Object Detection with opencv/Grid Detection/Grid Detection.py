import cv2
import matplotlib.pyplot as plt


# -------------------------ChessBoard Grid Patterns----------------------------
flat_chess = cv2.imread('flat_chessboard.png')
found, corners = cv2.findChessboardCorners(flat_chess,(7,7))
flat_chess_copy = flat_chess.copy()
cv2.drawChessboardCorners(flat_chess_copy, (7, 7), corners, found)
plt.imshow(flat_chess_copy)

# -------------------------Circle Based Grid Pattens---------------------------
dots = cv2.imread('dot_grid.png')
found, corners = cv2.findCirclesGrid(dots, (10,10), cv2.CALIB_CB_SYMMETRIC_GRID)
dbg_image_circles = dots.copy()
cv2.drawChessboardCorners(dbg_image_circles, (10, 10), corners, found)
plt.imshow(dbg_image_circles)