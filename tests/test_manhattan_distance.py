from unittest import TestCase

import numpy as np

from src.clustering import manhattan_distance


class TestManhattanDistance(TestCase):

    __THRESHOLD = 5

    def test_same_line(self):
        box_a = np.array([ 0, 0, 10, 10])
        box_b = np.array([20, 0, 30, 10])
        expected_distance = 10

        value_a_b = manhattan_distance(box_a, box_b, threshold=self.__THRESHOLD)
        value_b_a = manhattan_distance(box_b, box_a, threshold=self.__THRESHOLD)

        self.assertEqual(value_b_a, value_a_b)
        self.assertEqual(expected_distance, value_b_a)

    def test_above_below(self):
        box_a = np.array([0,  0, 10, 10])
        box_b = np.array([0, 20, 10, 30])
        expected_distance = 10

        value_a_b = manhattan_distance(box_a, box_b, threshold=self.__THRESHOLD)
        value_b_a = manhattan_distance(box_b, box_a, threshold=self.__THRESHOLD)

        self.assertEqual(value_a_b, value_b_a)
        self.assertEqual(expected_distance, value_a_b)

    def test_overlap_same_line(self):
        box_a = np.array([0, 0, 10, 10])
        box_b = np.array([5, 0, 15, 10])
        expected_distance = 0

        value_a_b = manhattan_distance(box_a, box_b, threshold=self.__THRESHOLD)
        value_b_a = manhattan_distance(box_a, box_b, threshold=self.__THRESHOLD)

        self.assertEqual(value_a_b, value_b_a)
        self.assertEqual(expected_distance, value_a_b)

    def test_overlap_above_below(self):
        box_a = np.array([0, 0, 10, 10])
        box_b = np.array([0, 5, 10, 15])
        expected_distance = 0

        value_a_b = manhattan_distance(box_a, box_b, threshold=self.__THRESHOLD)
        value_b_a = manhattan_distance(box_a, box_b, threshold=self.__THRESHOLD)

        self.assertEqual(value_a_b, value_b_a)
        self.assertEqual(expected_distance, value_a_b)

    def test_diagonal_overlap(self):
        box_a = np.array([0, 0, 10, 10])
        box_b = np.array([5, 5, 15, 15])
        expected_distance = 0

        value_a_b = manhattan_distance(box_a, box_b, threshold=self.__THRESHOLD)
        value_b_a = manhattan_distance(box_a, box_b, threshold=self.__THRESHOLD)

        self.assertEqual(value_a_b, value_b_a)
        self.assertEqual(expected_distance, value_a_b)

    def test_diagonal_touch1(self):
        box_a = np.array([ 0,  0, 10, 10])
        box_b = np.array([10, 10, 15, 15])
        expected_distance = 0

        value_a_b = manhattan_distance(box_a, box_b, threshold=self.__THRESHOLD)
        value_b_a = manhattan_distance(box_a, box_b, threshold=self.__THRESHOLD)

        self.assertEqual(value_a_b, value_b_a)
        self.assertEqual(expected_distance, value_a_b)

    def test_diagonal_touch2(self):
        box_a = np.array([20,  0, 30, 10])
        box_b = np.array([10, 10, 20, 20])
        expected_distance = 0

        value_a_b = manhattan_distance(box_a, box_b, threshold=self.__THRESHOLD)
        value_b_a = manhattan_distance(box_a, box_b, threshold=self.__THRESHOLD)

        self.assertEqual(value_a_b, value_b_a)
        self.assertEqual(expected_distance, value_a_b)

    def test_inside(self):
        box_a = np.array([20, 20, 50, 50])
        box_b = np.array([30, 30, 40, 40])
        expected_distance = 0

        value_a_b = manhattan_distance(box_a, box_b, threshold=self.__THRESHOLD)
        value_b_a = manhattan_distance(box_a, box_b, threshold=self.__THRESHOLD)

        self.assertEqual(value_a_b, value_b_a)
        self.assertEqual(expected_distance, value_a_b)

    def test_small_overlap_below(self):
        box_a = np.array([0,  0, 10, 10])
        box_b = np.array([0,  8,  8, 12])
        expected_distance = 0

        value_a_b = manhattan_distance(box_a, box_b, threshold=self.__THRESHOLD)
        value_b_a = manhattan_distance(box_a, box_b, threshold=self.__THRESHOLD)

        self.assertEqual(value_a_b, value_b_a)
        self.assertEqual(expected_distance, value_a_b)

    def test_self(self):
        box_a = np.array([20, 20, 50, 50])
        expected_distance = 0

        value_a_b = manhattan_distance(box_a, box_a, threshold=self.__THRESHOLD)

        self.assertEqual(expected_distance, value_a_b)
