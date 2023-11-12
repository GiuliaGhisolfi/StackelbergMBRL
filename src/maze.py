import numpy as np
import random

class Maze:
    #TODO: da sistemare commenti + documentazione
    """
    Generate a maze See https://en.wikipedia.org/wiki/Maze_generation_algorithm
    Returns either
        a (y, x, 2) size Numpy Array with 0 as a passage and 1 as a wall for the down and right walls of each cell; 
                    outer edges are always walls.
        a (y * 2 + 1, x * 2 + 1) size Numpy Array with 0 as a corridor and 1 as a wall block; outer edges are wall blocks.
    """

    def __init__(self, size_x, size_y):
        # initialize an array of walls filled with ones, data "not visited" + if exists for 2 walls (down and right) per cell.
        self.wall_size = np.array([size_y, size_x], dtype=np.int64)
        # add top, bottom, left, and right (ie. + 2 and + 2) to array size so can later work without checking going over its boundaries.
        self.walls = np.ones((self.wall_size[0] + 2, self.wall_size[1] + 2, 3), dtype=np.byte)
        # mark edges as "unusable" (-1)
        self.walls[:, 0, 0] = -1
        self.walls[:, self.wall_size[1] + 1, 0] = -1
        self.walls[0, :, 0] = -1
        self.walls[self.wall_size[0] + 1, :, 0] = -1

        # initialize an array of block data - each passage (0) and wall (1) is a block
        self.block_size = np.array([size_y * 2 + 1, size_x * 2 + 1], dtype=np.int64)
        self.blocks = np.ones((self.block_size[0], self.block_size[1]), dtype=np.byte)

        # generate walls
        self.gen_maze_2D()

    def gen_maze_walls(self, corridor_len=999):

        # Generate a maze.
        # This will start with a random cell and create a corridor until corridor length >= corridor_len or it cannot continue the current corridor.
        # Then it will continue from a random point within the current maze (ie. visited cells), creating a junction, until no valid starting points exist.
        # Setting a small corridor maximum length (corridor_len) will cause more branching / junctions.
        # Returns maze walls data - a NumPy array size (y, x, 2) with 0 or 1 for down and right cell walls.

        # set a random starting cell and mark it as "visited"
        cell = np.array([random.randrange(2, self.wall_size[0]), random.randrange(2, self.wall_size[1])], dtype=np.int64)
        self.walls[cell[0], cell[1], 0] = 0

        # a simple definition of the four neighboring cells relative to current cell
        up    = np.array([-1,  0], dtype=np.int64)
        down  = np.array([ 1,  0], dtype=np.int64)
        left  = np.array([ 0, -1], dtype=np.int64)
        right = np.array([ 0,  1], dtype=np.int64)

        # preset some variables
        need_cell_range = False
        round_nr = 0
        corridor_start = 0
        if corridor_len <= 4:
            corridor_len = 5  # even this is too small usually

        while np.size(cell) > 0:

            round_nr += 1
            # get the four neighbors for current cell (cell may be an array of cells)
            cell_neighbors = np.vstack((cell + up, cell + left, cell + down, cell + right))
            # valid neighbors are the ones not yet visited
            valid_neighbors = cell_neighbors[self.walls[cell_neighbors[:, 0], cell_neighbors[:, 1], 0] == 1]

            if np.size(valid_neighbors) > 0:
                # there is at least one valid neighbor, pick one of them (at random)
                neighbor = valid_neighbors[random.randrange(0, np.shape(valid_neighbors)[0]), :]
                if np.size(cell) > 2:
                    # if cell is an array of cells, pick one cell with this neighbor only, at random
                    cell = cell[np.sum(abs(cell - neighbor), axis=1) == 1]  # cells where distance to neighbor == 1
                    cell = cell[random.randrange(0, np.shape(cell)[0]), :]
                # mark neighbor visited
                self.walls[neighbor[0], neighbor[1], 0] = 0
                # remove the wall between current cell and neighbor. Applied to down and right walls only so may be that of the cell or the neighbor
                self.walls[min(cell[0], neighbor[0]), min(cell[1], neighbor[1]), 1 + abs(neighbor[1] - cell[1])] = 0
                # check if more corridor length is still available
                if round_nr - corridor_start < corridor_len:
                    # continue current corridor: set current cell to neighbor
                    cell = np.array([neighbor[0], neighbor[1]], dtype=np.int64)
                else:
                    # maximum corridor length fully used; make a new junction and continue from there
                    need_cell_range = True

            else:
                # no valid neighbors for this cell
                if np.size(cell) > 2:
                    # if cell already contains an array of cells, no more valid neighbors are available at all
                    cell = np.zeros((0, 0))  # this will end the while loop, the maze is finished.
                else:
                    # a dead end; make a new junction and continue from there
                    need_cell_range = True

            if need_cell_range:
                # get all visited cells (=0) not marked as "no neighbors" (=-1), start a new corridor from one of these (make a junction)
                cell = np.transpose(np.nonzero(self.walls[1:-1, 1:-1, 0] == 0)) + 1  # not checking the edge cells, hence needs the "+ 1"
                # check these for valid neighbors (any adjacent cell with "1" as visited status (ie. not visited) is sufficient, hence MAX)
                valid_neighbor_exists = np.array([self.walls[cell[:, 0] - 1, cell[:, 1], 0],
                                                  self.walls[cell[:, 0] + 1, cell[:, 1], 0],
                                                  self.walls[cell[:, 0], cell[:, 1] - 1, 0],
                                                  self.walls[cell[:, 0], cell[:, 1] + 1, 0]
                                                  ]).max(axis=0)
                # get all visited cells with no neighbors
                cell_no_neighbors = cell[valid_neighbor_exists != 1]
                # mark these (-1 = no neighbors) so they will no longer be actively used. This is not required but helps with large mazes.
                self.walls[cell_no_neighbors[:, 0], cell_no_neighbors[:, 1], 0] = -1
                corridor_start = round_nr + 0  # start a new corridor.
                need_cell_range = False

        # return: drop out the additional edge cells. All cells visited anyway so just return the down and right edge data.
        return self.walls[1:-1, 1:-1, 1:3]

    def gen_maze_2D(self, corridor_len=999):
        # converts walls data from gen_maze_walls to a NumPy array size (y * 2 + 1, x * 2 + 1)
        # wall blocks are represented by 1 and corridors by 0.

        self.gen_maze_walls(corridor_len)

        # use wall data to set final output maze
        self.blocks[1:-1:2, 1:-1:2] = 0  # every cell is visited if correctly generated
        # horizontal walls
        self.blocks[1:-1:2, 2:-2:2] = self.walls[1:-1, 1:-2, 2]  # use the right wall
        # vertical walls
        self.blocks[2:-2:2, 1:-1:2] = self.walls[1:-2, 1:-1, 1]  # use the down wall

        return self.blocks

