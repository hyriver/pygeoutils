from __future__ import annotations

import numpy as np
from shapely import LineString, MultiLineString

from pygeoutils import anchored_smoothing, smooth_linestring, smooth_multilinestring


def assert_close(a: float, b: float, rtol: float = 1e-3) -> bool:
    assert np.allclose(a, b, rtol=rtol)


class TestAnchoredSmoothing:
    def setup_method(self):
        self.line = LineString([(0, 0), (1, 1), (2, 0), (3, 1)])

    def test_simple_line(self):
        smoothed_line = anchored_smoothing(self.line, npts=10, sigma=1.0)
        assert isinstance(smoothed_line, LineString)
        assert len(smoothed_line.coords) == 10
        assert_close(smoothed_line.coords[0], (0, 0))
        assert_close(smoothed_line.coords[-1], (3, 1))

    def test_no_sigma(self):
        smoothed_line_no_sigma = anchored_smoothing(self.line, npts=10)
        assert isinstance(smoothed_line_no_sigma, LineString)
        assert len(smoothed_line_no_sigma.coords) == 10
        assert_close(smoothed_line_no_sigma.coords[0], (0, 0))
        assert_close(smoothed_line_no_sigma.coords[-1], (3, 1))

    def test_default_npts(self):
        smoothed_line_default_npts = anchored_smoothing(self.line)
        assert isinstance(smoothed_line_default_npts, LineString)
        assert len(smoothed_line_default_npts.coords) == len(self.line.coords)
        assert_close(smoothed_line_default_npts.coords[0], (0, 0))
        assert_close(smoothed_line_default_npts.coords[-1], (3, 1))

    def test_complex_line(self):
        complex_line = LineString([(0, 0), (1, 2), (2, 1), (3, 3), (4, 0)])
        smoothed_complex_line = anchored_smoothing(complex_line, npts=20, sigma=2.0)
        assert isinstance(smoothed_complex_line, LineString)
        assert len(smoothed_complex_line.coords) == 20
        assert_close(smoothed_complex_line.coords[0], (0, 0))
        assert_close(smoothed_complex_line.coords[-1], (4, 0))


class TestSmoothMultiLineString:
    def setup_method(self):
        self.mline = MultiLineString(
            [LineString([(0, 0), (1, 1), (2, 0)]), LineString([(2, 0), (3, 1), (4, 0)])]
        )

    def test_simple_multilinestring(self):
        smoothed_mline = smooth_multilinestring(self.mline, npts_list=[10, 10], sigma=1.0)
        assert isinstance(smoothed_mline, MultiLineString)
        assert len(smoothed_mline.geoms) == 2
        assert all(isinstance(line, LineString) for line in smoothed_mline.geoms)
        assert all(len(line.coords) == 10 for line in smoothed_mline.geoms)
        assert_close(smoothed_mline.geoms[0].coords[0], (0, 0))
        assert_close(smoothed_mline.geoms[0].coords[-1], (2, 0))
        assert_close(smoothed_mline.geoms[1].coords[0], (2, 0))
        assert_close(smoothed_mline.geoms[1].coords[-1], (4, 0))

    def test_no_sigma(self):
        smoothed_mline_no_sigma = smooth_multilinestring(self.mline, npts_list=[10, 10])
        assert isinstance(smoothed_mline_no_sigma, MultiLineString)
        assert len(smoothed_mline_no_sigma.geoms) == 2
        assert all(isinstance(line, LineString) for line in smoothed_mline_no_sigma.geoms)
        assert all(len(line.coords) == 10 for line in smoothed_mline_no_sigma.geoms)
        assert_close(smoothed_mline_no_sigma.geoms[0].coords[0], (0, 0))
        assert_close(smoothed_mline_no_sigma.geoms[0].coords[-1], (2, 0))
        assert_close(smoothed_mline_no_sigma.geoms[1].coords[0], (2, 0))
        assert_close(smoothed_mline_no_sigma.geoms[1].coords[-1], (4, 0))

    def test_default_npts_list(self):
        smoothed_mline_default_npts = smooth_multilinestring(self.mline)
        assert isinstance(smoothed_mline_default_npts, MultiLineString)
        assert len(smoothed_mline_default_npts.geoms) == 2
        assert all(isinstance(line, LineString) for line in smoothed_mline_default_npts.geoms)
        assert all(
            len(line.coords) == len(orig_line.coords)
            for line, orig_line in zip(smoothed_mline_default_npts.geoms, self.mline.geoms)
        )
        assert_close(smoothed_mline_default_npts.geoms[0].coords[0], (0, 0))
        assert_close(smoothed_mline_default_npts.geoms[0].coords[-1], (2, 0))
        assert_close(smoothed_mline_default_npts.geoms[1].coords[0], (2, 0))
        assert_close(smoothed_mline_default_npts.geoms[1].coords[-1], (4, 0))

    def test_complex_multilinestring(self):
        complex_mline = MultiLineString(
            [LineString([(0, 0), (1, 2), (2, 1)]), LineString([(2, 1), (3, 3), (4, 0)])]
        )
        smoothed_complex_mline = smooth_multilinestring(
            complex_mline, npts_list=[20, 20], sigma=2.0
        )
        assert isinstance(smoothed_complex_mline, MultiLineString)
        assert len(smoothed_complex_mline.geoms) == 2
        assert all(isinstance(line, LineString) for line in smoothed_complex_mline.geoms)
        assert all(len(line.coords) == 20 for line in smoothed_complex_mline.geoms)
        assert_close(smoothed_complex_mline.geoms[0].coords[0], (0, 0))
        assert_close(smoothed_complex_mline.geoms[0].coords[-1], (2, 1))
        assert_close(smoothed_complex_mline.geoms[1].coords[0], (2, 1))
        assert_close(smoothed_complex_mline.geoms[1].coords[-1], (4, 0))


class TestSmoothLineString:
    def setup_method(self):
        self.line = LineString([(0, 0), (1, 1), (2, 0), (3, 1)])

    def test_simple_line(self):
        smoothed_line = smooth_linestring(self.line, smoothing=1.0, npts=10)
        assert isinstance(smoothed_line, LineString)
        assert len(smoothed_line.coords) == 10
        assert_close(smoothed_line.coords[0], (0, 0))
        assert_close(smoothed_line.coords[-1], (3, 1))

    def test_no_smoothing(self):
        smoothed_line_no_smoothing = smooth_linestring(self.line, npts=10)
        assert isinstance(smoothed_line_no_smoothing, LineString)
        assert len(smoothed_line_no_smoothing.coords) == 10
        assert_close(smoothed_line_no_smoothing.coords[0], (0, 0))
        assert_close(smoothed_line_no_smoothing.coords[-1], (3, 1))

    def test_default_npts(self):
        smoothed_line_default_npts = smooth_linestring(self.line)
        assert isinstance(smoothed_line_default_npts, LineString)
        assert len(smoothed_line_default_npts.coords) == 20  # 5 times the number of original points
        assert_close(smoothed_line_default_npts.coords[0], (0, 0))
        assert_close(smoothed_line_default_npts.coords[-1], (3, 1))

    def test_complex_line(self):
        complex_line = LineString([(0, 0), (1, 2), (2, 1), (3, 3), (4, 0)])
        smoothed_complex_line = smooth_linestring(complex_line, smoothing=2.0, npts=20)
        assert isinstance(smoothed_complex_line, LineString)
        assert len(smoothed_complex_line.coords) == 20
        assert_close(smoothed_complex_line.coords[0], (-0.01685, 0.1429))
        assert_close(smoothed_complex_line.coords[-1], (3.993, 0.0590))
