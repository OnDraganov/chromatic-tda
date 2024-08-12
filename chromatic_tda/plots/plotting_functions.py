import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from chromatic_tda.entities.simplicial_complex import SimplicialComplex
from chromatic_tda.utils.floating_point_utils import FloatingPointUtils


def plot_persistence_diagram(bars, ax=None, **kwargs):
    """Plot a single persistence diagram. Bars can be given in several different formats:
        - list of bars, i.e., (birth, death) tuples
        - list of (dimension, (birth, death)) tuples ... prints each dimension with different marker/color
        - dictionary of the form { dimension : list_of_bars }
    A matplotlib axis can be given as the ax argument.
    If keyword argument 'max_dimension' is a number, only print diagrams up to that dimension.
    Various options can be given via keyword arguments, many coming from matplotlib options. Include the following,
    with default values in parentheses:
        size (5), facecolor ('white'), aspect ('equal'), 'lim' (None), markersize (6), fillstyle ('none'), alpha (.7)
        ticks_fontsize (9), label_fontsize (11), title_fontsize (17), legend_fontsize (11)
        axes_labels (True), only_finite (False), label_inf (True), infinite_line_legend (True),
        color ({0: 'tab:green', 1: 'blue', 2: 'black', 3: 'orange'}), marker ({0: 'D', 1: 'o', 2: 's', 3: '*'}),
    """
    bars_dict = PlottingUtils().process_input_to_dict_form(bars)
    if 'max_dimension' in kwargs:
        bars_dict = {dim : bars for dim, bars in bars_dict.items() if dim <= kwargs['max_dimension']}
    bars_dict_fin, bars_dict_inf = PlottingUtils().split_finite_infinite_dictionary(bars_dict)

    if 'lim' in kwargs and kwargs['lim'] is not None:
        xlim, ylim = PlottingUtils().parse_plot_limit_argument(kwargs['lim'])
    else:  # is not in kwargs (or is None)
        xlim, ylim = PlottingUtils().find_plot_limits(bars_dict_fin, **kwargs)

    infinity_position = kwargs.get('infinity_position', ylim[1] * .98)

    if ax:
        plt.sca(ax)
    else:
        plt.figure(figsize=(kwargs.get('size', 5), kwargs.get('size', 5)),
                   facecolor=kwargs.get('facecolor', 'white'))
    if not FloatingPointUtils.is_close(*xlim):  # to avoid passing identical low and high xlim to pyplot
        plt.xlim(xlim)
    if not FloatingPointUtils.is_close(*ylim):  # to avoid passing identical low and high ylim to pyplot
        plt.ylim(ylim)
    plt.gca().set_aspect(kwargs.get('aspect', 'equal'))
    plt.xticks(fontsize=kwargs.get('ticks_fontsize', 9))
    plt.yticks(fontsize=kwargs.get('ticks_fontsize', 9))
    if kwargs.get('axes_labels', True):
        plt.xlabel('birth', fontsize=kwargs.get('label_fontsize', 11))
        plt.ylabel('death', fontsize=kwargs.get('label_fontsize', 11))

    color = kwargs.get('color', {0: 'tab:green', 1: 'blue', 2: 'black', 3: 'orange'})
    marker = kwargs.get('marker', {0: 'D', 1: 'o', 2: 's', 3: '*'})
    if type(color) == list:
        color = {d: color[i] for i, d in enumerate(sorted(bars_dict)) if len(color) > i}
    elif type(color) == str:
        color = {d: color for d in bars_dict}

    if type(marker) == list:
        marker = {d: marker[i] for i, d in enumerate(sorted(bars_dict)) if len(marker) > i}
    elif type(marker) == str:
        marker = {d: marker for d in bars_dict}

    plt.axvline(0, color='gray', linewidth=1, alpha=.5)
    plt.plot(xlim, xlim, color='black', linewidth=1, alpha=.5)
    if 'title' in kwargs:
        plt.title(kwargs['title'], fontsize=kwargs.get('title_fontsize', 17))
    if not kwargs.get('only_finite', False) and any(len(bars_inf) > 0 for bars_inf in bars_dict_inf.values()):  # if also plotting infinite
        label = 'inf' if kwargs.get('label_inf', True) else None
        plt.axhline(infinity_position, color='tab:red', alpha=.5, linewidth=1, label=label)
        if label is not None:
            kwargs['infinite_line_legend'] = True
    for d in sorted(bars_dict_fin):
        if kwargs.get('only_finite', False):
            bars = bars_dict_fin[d]
        else:
            bars = list(bars_dict_fin[d]) + [[birth, infinity_position] for birth, death in bars_dict_inf[d]]
        plt.plot(
            *zip(*bars),
            linestyle='none',
            markersize=kwargs.get('markersize', 6),
            marker=marker.get(d, 'o'),
            color=color.get(d, None),
            fillstyle=kwargs.get('fillstyle', 'none'),
            alpha=kwargs.get('alpha', .7),
            label=str(d) if d != '' else None
        )
    if len(set(bars_dict_fin.keys()) - {''}) > 0 or kwargs.get('infinite_line_legend', False):
        plt.legend(loc='center right', fontsize=kwargs.get('legend_fontsize', 11))

    return plt.gcf(), plt.gca()


def plot_six_pack(data, axs=None, **kwargs):
    """Plot a six-pack of persistence diagrams. Data can be given in several different formats:
        - SimplicialComplex object
        - Dictionary with keys 'kernel', 'relative', 'cokernel', 'sub_complex', 'image', 'complex', and values
          the bars for the given group. The bars can be given in the following formats:
            - list of bars, i.e., (birth, death) tuples
            - list of (dimension, (birth, death)) tuples ... prints each dimension with different marker/color
            - dictionary of the form { dimension : list_of_bars }
    A list of six matplotlib axes can be given as the axs argument.
    Various options can be given via keyword arguments, many coming from matplotlib options. For more details see
    docstring of plot_persistence_diagram.
    If keyword argument 'max_dimension' is a number, only print diagrams up to that dimension.
    """
    if isinstance(data, SimplicialComplex):
        data = data.bars_six_pack()
    if isinstance(data, dict):
        data = {group : PlottingUtils().process_input_to_dict_form(bars) for group, bars in data.items()}
    else:
        raise TypeError("The data to plot a six-pack need to be given either as SimplicialComplex or a dictionary.")

    if 'max_dimension' in kwargs:
        for group in data:
            data[group] = {dim : bars for dim, bars in data[group].items() if dim <= kwargs['max_dimension']}

    if axs is None:
        fig, axs = plt.subplots(2, 3,
                                figsize=(3 * kwargs.get('size', 5), 2 * kwargs.get('size', 5)),
                                facecolor=kwargs.get('facecolor', 'white'))
    axs = np.array(axs).flatten()
    if 'lim' in kwargs and kwargs['lim'] is not None:
        xlim, ylim = PlottingUtils().parse_plot_limit_argument(kwargs['lim'])
    else:
        xlims, ylims = zip(*[PlottingUtils().find_plot_limits(bars_dict) for group, bars_dict in data.items()])
        xlim = (min(x[0] for x in xlims), max(x[1] for x in xlims))
        ylim = (min(y[0] for y in ylims), max(y[1] for y in ylims))
    kwargs['lim'] = (xlim, ylim)

    if kwargs.get('axes_labels', True):
        plt.gcf().supxlabel('birth', fontsize=kwargs.get('label_fontsize', 11), y=.06)
        plt.gcf().supylabel('death', fontsize=kwargs.get('label_fontsize', 11), x=.09)
        kwargs['axes_labels'] = False

    groups_order = ['kernel', 'relative', 'cokernel', 'sub_complex', 'image', 'complex']
    if kwargs.get('allow_arbitrary_keys', False):
        groups = sorted(data.keys(), key=lambda k: groups_order.index(k) if k in groups_order else len(groups_order))
    else:
        groups = groups_order
        if set(groups) != set(data.keys()):
            raise ValueError("Keys in the data dictionary need to be 'kernel', 'relative', 'cokernel', 'sub_complex', "
                             "'image', 'complex'. To allow different keys, pass allow_arbitrary_keys=True.")

    for group, ax in zip(groups, axs):
        plot_persistence_diagram(data[group], ax=ax, title=group.replace('_', '-'), **kwargs)

    return plt.gcf(), axs


def plot_labeled_point_set(points, labels, ax=None, **kwargs):
    """A simple plot of points and labels"""
    if ax:
        plt.sca(ax)
    else:
        plt.figure(figsize=(kwargs.get('size', 7), kwargs.get('size', 7)),
                   facecolor=kwargs.get('facecolor', 'white'))
    if len(points) > 0 and len(points[0]) != 2:
        raise ValueError('Only two-dimensional point sets can be plotted.')
    colors = kwargs.get('colors', {})
    markers = kwargs.get('markers', {0: 'o', 1: 's', 2: '^'})
    for i, current_label in enumerate(sorted(set(labels))):
        plt.plot(*zip(*[pt for pt, lab in zip(points, labels) if lab == current_label]),
                 linestyle='none',
                 marker=markers.get(i, 'o'),
                 color=colors.get(i, None),
                 label=str(current_label),
                 alpha=kwargs.get('alpha', 1))
    plt.gca().set_aspect('equal')
    if kwargs.get('legend', False):
        plt.legend()
    return plt.gcf(), plt.gca()


class PlottingUtils:

    X_AXIS_EXTRA_RELATIVE_SPACE = 0.03
    Y_AXIS_EXTRA_RELATIVE_SPACE = 0.02
    Y_AXIS_EXTRA_RELATIVE_SPACE_WITH_INFINITY = 0.07

    def __init__(self):
        pass

    @staticmethod
    def find_max_birth_single_diagram(bars) -> float:
        return max((birth for birth, death in bars), default=0)

    @staticmethod
    def find_max_birth_dictionary(bars_dict) -> float:
        return max((PlottingUtils.find_max_birth_single_diagram(bars) for bars in bars_dict.values()), default=0)

    @staticmethod
    def find_max_finite_death_single_diagram(bars) -> float:
        return max((death for birth, death in bars if death < float('inf')), default=0)

    @staticmethod
    def find_max_finite_death_dictionary(bars_dict) -> float:
        return max((PlottingUtils.find_max_finite_death_single_diagram(bars) for bars in bars_dict.values()), default=0)

    @staticmethod
    def find_plot_limits(bars_dict, **kwargs):
        maxdeath = PlottingUtils.find_max_finite_death_dictionary(bars_dict)
        maxbirthdeath = max(maxdeath, PlottingUtils.find_max_birth_dictionary(bars_dict))
        if kwargs.get('only_finite', False):
            lim_left = -PlottingUtils.X_AXIS_EXTRA_RELATIVE_SPACE * maxdeath
            xlim = (lim_left, maxdeath * (1 + PlottingUtils.Y_AXIS_EXTRA_RELATIVE_SPACE))
            ylim = (lim_left, maxdeath * (1 + PlottingUtils.Y_AXIS_EXTRA_RELATIVE_SPACE))
        else:
            lim_left = -PlottingUtils.X_AXIS_EXTRA_RELATIVE_SPACE * maxbirthdeath
            xlim = (lim_left, maxbirthdeath * (1 + PlottingUtils.Y_AXIS_EXTRA_RELATIVE_SPACE))
            ylim = (lim_left, maxbirthdeath * (1 + PlottingUtils.Y_AXIS_EXTRA_RELATIVE_SPACE_WITH_INFINITY))

        return xlim, ylim

    @staticmethod
    def parse_plot_limit_argument(lim) -> tuple[tuple[float, float], tuple[float, float]]:
        if isinstance(lim, list) or isinstance(lim, tuple):
            if isinstance(lim[0], list) or isinstance(lim[0], tuple):
                xlim = lim[0]
                ylim = lim[1]
            else:  # is (xmax, ymax) tuple
                lim_left = -PlottingUtils.X_AXIS_EXTRA_RELATIVE_SPACE * max(lim[0], lim[1])
                xlim = (lim_left, lim[0])
                ylim = (lim_left, lim[1])
        else:  # is just one number
            lim_left = -PlottingUtils.X_AXIS_EXTRA_RELATIVE_SPACE * lim
            xlim = (lim_left, lim)
            ylim = (lim_left, lim)

        return xlim, ylim

    @staticmethod
    def convert_tuple_form_to_dict_form(bars_tuple_form):
        bars_dict_form = {}
        for dim, bar in bars_tuple_form:
            if dim not in bars_dict_form:
                bars_dict_form[dim] = []
            bars_dict_form[dim].append(bar)
        return bars_dict_form

    @staticmethod
    def process_input_to_dict_form(input_form):
        if len(input_form) == 0:
            return {}
        if isinstance(input_form, dict):
            return input_form
        list_form = list(input_form)
        if len(list_form[0]) == 2:
            if isinstance(list_form[0][0], int):
                try:
                    second_coordinate_length = len(list_form[0][1])
                except TypeError:
                    second_coordinate_length = -1
                if second_coordinate_length == 2:
                    return PlottingUtils.convert_tuple_form_to_dict_form(list_form)
            return {'' : list_form}
        raise TypeError("Bars given in an invalid format. Should be either a dim -> list_of_bars dictionary"
                        " or a (dim, (birth, death)) list.")

    @staticmethod
    def split_finite_infinite_single_diagram(bars_list):
        return ([bar for bar in bars_list if bar[1] < float('inf')],
                [bar for bar in bars_list if bar[1] == float('inf')])

    @staticmethod
    def split_finite_infinite_dictionary(bars_dict):
        bars_finite = {}
        bars_infinite = {}
        for dim in bars_dict:
            bars_finite[dim], bars_infinite[dim] = PlottingUtils.split_finite_infinite_single_diagram(bars_dict[dim])
        return bars_finite, bars_infinite

    @staticmethod
    def add_counts_text_to_markers(axs, rtol=1e-5, atol=1e-8):
        """Given axes, counts markers close to each other and adds the counts texts"""
        if isinstance(axs, Axes):
            axs = [axs]
        axs = np.array(axs).flatten()

        xlim, ylim = axs[0].get_xlim()[1] - axs[0].get_xlim()[0], axs[0].get_ylim()[1] - axs[0].get_ylim()[0]

        for ax in axs.flatten():
            for child in ax.get_children():
                if str(child.get_label()).isnumeric():
                    # extract points with frequencies
                    data = {}
                    for point in zip(*child.get_data()):
                        for key in data:
                            if (np.isclose(point[0], key[0], rtol=rtol, atol=atol)
                                    and np.isclose(point[1], key[1], rtol=rtol, atol=atol)):
                                data[key] += 1
                                break
                        else:
                            data[point] = 1
                    # print texts
                    for point, frequency in data.items():
                        if point[0] < .0 or point[1] > .9 * ylim:
                            posx = point[0] + .01 * xlim if point[0] < .9 * xlim else point[0] - .035 * xlim
                            posy = point[1] - .04 * ylim
                        else:
                            posx = point[0] + .005 * xlim
                            posy = point[1] + .015 * ylim
                        ax.text(posx, posy, frequency, fontdict={'color': child.get_color()})
