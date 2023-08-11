import numpy as np
import multiprocessing
import math

from chromatic_tda.utils.singleton import singleton
from chromatic_tda.core.core_chromatic_alpha_complex import CoreChromaticAlphaComplex
from chromatic_tda.utils.geometrical_utils import GeometricalUtils
from chromatic_tda.core.core_simplicial_complex import CoreSimplicialComplex
from chromatic_tda.utils.timing import TimingUtils


class RadiusFunctionParallelUtils():

    atoms : int
    alpha_complex : CoreChromaticAlphaComplex
    simplex_list : list

    def __init__(self, alpha_complex) -> None:
        # we can use os.cpu_count()
        self.atoms= 5
        self.alpha_complex = alpha_complex

    def calculate_radius_function(self, simplex):
        dim = len(simplex) - 1

        if dim == 4:
            return RadiusFunctionUtils()._compute_radius_function_pentachoron(self.alpha_complex, simplex)
        elif dim == 3:
            return RadiusFunctionUtils()._compute_radius_function_tetrahedron(self.alpha_complex, simplex)
        elif dim == 2:
            return RadiusFunctionUtils()._compute_radius_function_triangle(self.alpha_complex, simplex)
        elif dim == 1:
            return RadiusFunctionUtils()._compute_radius_function_edge(self.alpha_complex, simplex)
        
        raise NotImplementedError()

    def calculate(self, atom):
        actions = math.ceil(len(self.simplex_list)/self.atoms)
        res = []
        for i in range(0, actions):
            k = atom * actions + i
            if k >= len(self.simplex_list):
                break
            simplex = self.simplex_list[k]
            rad = self.calculate_radius_function(simplex)
            res.append((simplex, rad))
        return res        

    def compute_radius_function_in_parallel(self, **kwargs) -> None:
        TimingUtils().start("Compute Radius Function Parallel")                    
        print("Parallel Processing Started")

        for dim in [4, 3, 2, 1]:
            size_of_process = len(self.alpha_complex.simplicial_complex.dim_simplex_dict.get(dim, {}))

            if size_of_process == 0:
                continue

            self.simplex_list = []
            for simplex in self.alpha_complex.simplicial_complex.dim_simplex_dict.get(dim, {}):
                self.simplex_list.append(simplex)

            with multiprocessing.Pool() as pool:
                res_list = pool.map(self.calculate, range(0, self.atoms))
            
                for res in res_list:
                    for item in res:
                        self.alpha_complex.sq_rad[item[0]] = item[1]

        if 'round' in kwargs:
            self.alpha_complex.simplicial_complex.set_simplex_weights(
                {s : np.round(np.sqrt(r), decimals=kwargs['round']) for s,r in self.alpha_complex.sq_rad.items()},
                default_value=0)
        else:
            self.alpha_complex.simplicial_complex.set_simplex_weights({s : np.sqrt(r) for s,r in self.alpha_complex.sq_rad.items()}, default_value = 0)
                        
        TimingUtils().stop("Compute Radius Function Parallel")


@singleton
class RadiusFunctionUtils():

    def compute_radius_function(self, alpha_complex: CoreChromaticAlphaComplex, **kwargs) -> None:
        TimingUtils().start("Compute Radius Function")

        for simplex in alpha_complex.simplicial_complex.dim_simplex_dict.get(4, {}):
            alpha_complex.sq_rad[simplex] = self._compute_radius_function_pentachoron(alpha_complex, simplex)
        for simplex in alpha_complex.simplicial_complex.dim_simplex_dict.get(3, {}):
            alpha_complex.sq_rad[simplex] = self._compute_radius_function_tetrahedron(alpha_complex, simplex)
        for simplex in alpha_complex.simplicial_complex.dim_simplex_dict.get(2, {}):
            alpha_complex.sq_rad[simplex] = self._compute_radius_function_triangle(alpha_complex, simplex)
        for simplex in alpha_complex.simplicial_complex.dim_simplex_dict.get(1, {}):
            alpha_complex.sq_rad[simplex] = self._compute_radius_function_edge(alpha_complex, simplex)
        if 'round' in kwargs:
            alpha_complex.simplicial_complex.set_simplex_weights(
                {s : np.round(np.sqrt(r), decimals=kwargs['round']) for s,r in alpha_complex.sq_rad.items()},
                default_value=0)
        else:
            alpha_complex.simplicial_complex.set_simplex_weights({s : np.sqrt(r) for s,r in alpha_complex.sq_rad.items()}, default_value = 0)

        TimingUtils().stop("Compute Radius Function")

    def _compute_radius_function_pentachoron(self, alpha_complex: CoreChromaticAlphaComplex, simplex):
        TimingUtils().start("Compute Radius Function :: Pentachoron")

        red, grn, blu = (alpha_complex.points[list(s)] for s in alpha_complex.split_simplex_sort_by_size(simplex))
        if (len(red), len(grn), len(blu)) == (2,2,1):
            E_pt = GeometricalUtils().intersect_lines(*GeometricalUtils().bisector(*red), *GeometricalUtils().bisector(*grn))
            res = GeometricalUtils().sq_dist_max(E_pt, red[0], grn[0], blu[0])
        elif (len(red), len(grn), len(blu)) == (3,1,1):
            E_pt = GeometricalUtils().circum_center(*red)
            res = GeometricalUtils().sq_dist_max(E_pt, red[0], grn[0], blu[0])
        else:
            raise RuntimeError(f"Simplex colored {len(red)}:{len(grn)}:{len(blu)} cannot occur in 2D.")

        TimingUtils().stop("Compute Radius Function :: Pentachoron")
        return res


    def _compute_radius_function_tetrahedron(self, alpha_complex: CoreChromaticAlphaComplex, simplex):
        TimingUtils().start("Compute Radius Function :: Tetrahedron")

        red, grn, blu = (alpha_complex.points[list(s)] for s in alpha_complex.split_simplex_sort_by_size(simplex))
        if (len(red), len(grn), len(blu)) == (3,1,0):
            E_pt = GeometricalUtils().circum_center(*red)
            if alpha_complex.is_empty_stack(E_pt, simplex):
                res = GeometricalUtils().sq_dist_max(E_pt, red[0], grn[0])
            else:
                res = min(alpha_complex.sq_rad[cf] for cf in alpha_complex.simplicial_complex.co_boundary[simplex])

        elif (len(red), len(grn), len(blu)) == (2,2,0):
            E_pt = GeometricalUtils().intersect_lines(*GeometricalUtils().bisector(*red), *GeometricalUtils().bisector(*grn))
            if alpha_complex.is_empty_stack(E_pt, simplex):
                res = GeometricalUtils().sq_dist_max(E_pt, red[0], grn[0])
            else:
                res = min(alpha_complex.sq_rad[cf] for cf in alpha_complex.simplicial_complex.co_boundary[simplex])

        elif (len(red), len(grn), len(blu)) == (2,1,1):
            p, p_dir = GeometricalUtils().bisector(*red)
            pts = [
                GeometricalUtils().project_to_line(p, p_dir, red[0]),
                GeometricalUtils().project_to_line(p, p_dir, grn[0]),
                GeometricalUtils().project_to_line(p, p_dir, blu[0]),
                GeometricalUtils().intersect_lines(p, p_dir, *GeometricalUtils().bisector(red[0], grn[0])),
                GeometricalUtils().intersect_lines(p, p_dir, *GeometricalUtils().bisector(red[0], blu[0])),
                GeometricalUtils().intersect_lines(p, p_dir, *GeometricalUtils().bisector(grn[0], blu[0]))
            ]
            radii = [max(   GeometricalUtils().sq_dist(pt, red[0]),
                            GeometricalUtils().sq_dist(pt, grn[0]),
                            GeometricalUtils().sq_dist(pt, blu[0])) for pt in pts]
            radius, center = min( (r,p) for r,p in zip(radii, pts))
            if alpha_complex.is_empty_stack(center, simplex):
                res = radius
            else:
                res = min(alpha_complex.sq_rad[cf] for cf in alpha_complex.simplicial_complex.co_boundary[simplex])
        else:
            raise RuntimeError(f"Simplex colored {len(red)}:{len(grn)}:{len(blu)} cannot occur in 2D.")

        TimingUtils().stop("Compute Radius Function :: Tetrahedron")
        return res

    def _compute_radius_function_triangle(self, alpha_complex: CoreChromaticAlphaComplex, simplex):
        TimingUtils().start("Compute Radius Function :: Triangle")

        red, grn, blu = (alpha_complex.points[list(s)] for s in alpha_complex.split_simplex_sort_by_size(simplex))

        if (len(red), len(grn), len(blu)) == (3,0,0):
            E_pt = GeometricalUtils().circum_center(*red)
            if alpha_complex.is_empty_stack(E_pt, simplex):
                res = GeometricalUtils().sq_dist_max(E_pt, red[0])
            else:
                res = min(alpha_complex.sq_rad[cf] for cf in alpha_complex.simplicial_complex.co_boundary[simplex])

        elif (len(red), len(grn), len(blu)) == (2,1,0):
            p, p_dir = GeometricalUtils().bisector(*red)
            
            pts = [
                GeometricalUtils().project_to_line(p, p_dir, red[0]),
                GeometricalUtils().project_to_line(p, p_dir, grn[0]),
                GeometricalUtils().intersect_lines(p, p_dir, *GeometricalUtils().bisector(red[0], grn[0]))
            ]
            radii = [max(   GeometricalUtils().sq_dist(pt, red[0]),
                            GeometricalUtils().sq_dist(pt, grn[0])) for pt in pts]
            radius, center = min( (r,p) for r,p in zip(radii, pts))
            if alpha_complex.is_empty_stack(center, simplex):
                res = radius
            else:
                res = min(alpha_complex.sq_rad[cf] for cf in alpha_complex.simplicial_complex.co_boundary[simplex])

        elif (len(red), len(grn), len(blu)) == (1,1,1):
            pts = [ # no need to check red[0],grn[0],blu[0], max rad is never maxed there
                GeometricalUtils().bisector(red[0], grn[0])[0],
                GeometricalUtils().bisector(red[0], blu[0])[0],
                GeometricalUtils().bisector(grn[0], blu[0])[0],
                GeometricalUtils().circum_center(red[0], grn[0], blu[0])
            ]
            radii = [max(   GeometricalUtils().sq_dist(pt, red[0]),
                            GeometricalUtils().sq_dist(pt, grn[0]),
                            GeometricalUtils().sq_dist(pt, blu[0])) for pt in pts]
            radius, center = min( (r,p) for r,p in zip(radii, pts))
            if alpha_complex.is_empty_stack(center, simplex):
                res = radius
            else:
                res = min(alpha_complex.sq_rad[cf] for cf in alpha_complex.simplicial_complex.co_boundary[simplex])
        else:
            raise RuntimeError(f"Simplex colored {len(red)}:{len(grn)}:{len(blu)} cannot occur in 2D.")

        TimingUtils().stop("Compute Radius Function :: Triangle")
        return res

    def _compute_radius_function_edge(self, alpha_complex: CoreChromaticAlphaComplex, simplex):
        TimingUtils().start("Compute Radius Function :: Edge")

        red, grn, blu = (alpha_complex.points[list(s)] for s in alpha_complex.split_simplex_sort_by_size(simplex))
        if (len(red), len(grn), len(blu)) == (2,0,0):
            E_pt, _ = GeometricalUtils().bisector(*red)
            if alpha_complex.is_empty_stack(E_pt, simplex):
                res = GeometricalUtils().sq_dist_max(E_pt, red[0])
            else:
                res = min(alpha_complex.sq_rad[cf] for cf in alpha_complex.simplicial_complex.co_boundary[simplex])
        elif (len(red), len(grn), len(blu)) == (1,1,0):
            E_pt, _ = GeometricalUtils().bisector(red[0], grn[0])
            if alpha_complex.is_empty_stack(E_pt, simplex):
                res = GeometricalUtils().sq_dist_max(E_pt, red[0], grn[0]) #grn[0] not needed here
            else:
                res = min(alpha_complex.sq_rad[cf] for cf in alpha_complex.simplicial_complex.co_boundary[simplex])
        else:
            raise RuntimeError(f"Simplex colored {len(red)}:{len(grn)}:{len(blu)} cannot occur in 2D.")

        TimingUtils().stop("Compute Radius Function :: Edge")
        return res

    def check_monotonicity_of_radius_function(self, complex: CoreSimplicialComplex):
        return all(all(complex.simplex_weights[b] <= complex.simplex_weights[simplex] for b in boundary) for simplex, boundary in complex.boundary.items())
