from statannotations.Annotator import Annotator
from itertools import product, combinations
from consts import MODEL_FANCY_ORDER, RESIST_EFFECT_MUTS, RESIST_NEUTRAL_MUTS, CORECEPTOR_EFFECT_MUTS, CORECEPTOR_NEUTRAL_MUTS
from scipy.special import logit
import yaml
import pandas as pd

def model_test_pairs(field, transpose=False, fancy=True):
    pairs = []
    models = MODEL_FANCY_ORDER if fancy else MODEL_ORDER
    if not transpose:
        for f in field:
            pairs += [[(f, m1), (f, m2)] for m1, m2 in combinations(models, 2)]
    else:
        for m in models:
            pairs += [[(m, f1), (m, f2)] for f1, f2 in combinations(field, 2)]
    return pairs


class FilteredAnnotator(Annotator):
    def annotate(self, line_offset=None, line_offset_to_group=None):
        """Add configured annotations to the plot."""
        self._check_has_plotter()

        self._maybe_warn_about_configuration()

        self._update_value_for_loc()

        ann_list = []
        orig_value_lim = self._plotter.get_value_lim()

        offset_func = self.get_offset_func(self.loc)
        self.value_offset, self.line_offset_to_group = offset_func(
            line_offset, line_offset_to_group
        )

        if self._verbose:
            self.print_pvalue_legend()

        ax_to_data = self._plotter.get_transform_func("ax_to_data")

        self.validate_test_short_name()

        for annotation in self.annotations:
            if annotation.data.pvalue < self._alpha:
                self._annotate_pair(
                    annotation,
                    ax_to_data=ax_to_data,
                    ann_list=ann_list,
                    orig_value_lim=orig_value_lim,
                )

        # reset transformation
        y_stack_max = max(self._value_stack_arr[1, :])
        ax_to_data = self._plotter.get_transform_func("ax_to_data")
        value_lims = (
            (
                [(0, 0), (0, max(1.04 * y_stack_max, 1))]
                if self.loc == "inside"
                else [(0, 0), (0, 1)]
            )
            if self.orient == "v"
            else (
                [(0, 0), (max(1.04 * y_stack_max, 1), 0)]
                if self.loc == "inside"
                else [(0, 0), (1, 0)]
            )
        )
        set_lims = self.ax.set_ylim if self.orient == "v" else self.ax.set_xlim
        transformed = ax_to_data.transform(value_lims)
        set_lims(transformed[:, 1 if self.orient == "v" else 0])

        return self._get_output()

    
def path2info(path):
    parts = path.split("/")
    if '-' in parts[-1]:
        parts = path.split("/")
        stem = parts[-3]
        model, prot, task = stem.split("_")
        fold = parts[-2]
        ds = int(parts[-1].split('-')[0])
        row_info = {"fold": fold, "model": model, "prot": prot, "task": task, 'DS': ds}
    else:
        stem = parts[-3]
        model, prot, task = stem.split("_")
        fold = parts[-2]
        row_info = {"fold": fold, "model": model, "prot": prot, "task": task, 'DS': 'Full'}
        
    with open(path) as handle:
        metrics = yaml.full_load(handle)
        
    return row_info, metrics

def parse_model_results(files):
    data = []
    for path in files:
        row_info, metrics = path2info(path)
        for key in metrics.keys():
            if "__" in key:
                field, metric = key.split("__")
                field = field.replace("test_", "")
                data.append(dict(row_info.items()))
                data[-1]["metric"] = metric
                data[-1]["field"] = field
                data[-1]["value"] = metrics[key]

    return pd.DataFrame(data)    


def process_mutation_dataset(dataset, effect_muts, neutral_muts, loss_only=True):
    e = 1e-6
    
    orig = pd.DataFrame(dataset["sampled"]).reset_index()
    
    fields = list(orig.columns)

    data = [pd.melt(orig, id_vars=['index'], var_name='field')]
    data[-1]['metric'] = 'WT'
    
    order = [(m, 'functional') for m in effect_muts]
    order += [(m, 'neutral') for m in neutral_muts]
    for mut_name, typ in order:
        mut = pd.DataFrame(dataset[mut_name])
        mut['index'] = orig['index']
        mut['metric'] = mut['change'].map(lambda c: f'{typ}-{c}-{mut_name}')
        data.append(pd.melt(mut[['metric']+fields],
                            id_vars=['index', 'metric'],
                            var_name='field'))

        delta = logit(mut[fields].clip(e, 1-e)) - logit(orig[fields].clip(e, 1-e))

        if typ == "functional":
            for field in fields:
                expected_mask = (delta[field] > 0) & (mut["change"] == "gain")
                expected_mask |= (delta[field] < 0) & (mut["change"] == "loss")
                delta.loc[expected_mask, field] = 0
            loss = delta**2
        else:

            loss = delta**2
        loss['metric'] = 'concept_loss'
        loss['index'] = orig['index']

        data.append(pd.melt(loss[['metric']+fields],
                            id_vars=['index', 'metric'],
                            var_name='field'))
    
    df = pd.concat(data, axis=0, ignore_index=True)
    
    if loss_only:
        closs = df.query('metric == "concept_loss"').drop(['index'], axis=1)
        ndf = closs.groupby('field', as_index=False)['value'].mean()
        ndf['metric'] = 'concept_loss'
        return ndf
    else:
        return df
    
def task2muts(task):
    if task == 'resist':
        return RESIST_EFFECT_MUTS, RESIST_NEUTRAL_MUTS
    elif task == 'coreceptor':
        return CORECEPTOR_EFFECT_MUTS, CORECEPTOR_NEUTRAL_MUTS