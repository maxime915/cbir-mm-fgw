import pathlib
import re

import pandas as pd
import yaml


def try_int(s: str):
    try:
        int(s)
        return s
    except ValueError:
        return None


def mem_to_gb(mem_str):
    match = re.match(r"([\d.]+)(TiB|GiB|MiB|KiB)", mem_str)
    if not match:
        raise ValueError(f"Invalid memory format: {mem_str}")

    value, unit = match.groups()
    value = float(value)

    factors = {
        "TiB": 1024.0,
        "GiB": 1.0,
        "MiB": 1.0 / 1024.0,
        "KiB": 1.0 / (1024.0 * 1024.0)
    }

    return value * factors[unit]

def time_to_seconds(time_str):
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def gather(source: pathlib.Path, filters: dict[str, list] | None = None):
    if filters is None:
        filters = {}

    contents = list(source.iterdir())

    # pair up all .out and .yaml files
    slurm_logs_f = {
        k: f
        for f in contents  # {JOB_IDX}_{k}_search.out
        if f.name.endswith("_search.out") and (k := f.name.split("_")[1]) != ""
    }

    configs_f = {
        k: f
        for f in contents  # {label}-{k}.yaml
        if f.suffix == ".yaml" and (k := try_int(f.stem.split("-")[-1])) is not None
    }

    columns = ("tile_size_um", "model", "tag", "top1", "top3", "top5", "qtime", "mem", "twtime")
    rows: list[tuple[float, str, str, float, float, float, float, float, float]] = []

    if sorted(slurm_logs_f.keys()) != sorted(configs_f.keys()):
        s_only = sorted(set(slurm_logs_f.keys()) - set(configs_f.keys()))
        c_only = sorted(set(configs_f.keys()) - set(slurm_logs_f.keys()))
        print(f"{len(slurm_logs_f)=} {s_only=}")
        print(f"{len(configs_f)=} {c_only=}")
        raise ValueError("mismatch in keys")

    for k, config_f in configs_f.items():
        log = slurm_logs_f[k].read_text("utf8")
        lines = log.splitlines()
        cfg = yaml.safe_load(config_f.read_text("utf8"))["base_config"]

        skip = False
        for f_k, f_vals in filters.items():
            if cfg.get(f_k) not in f_vals:
                skip = True
                break
        if skip:
            continue

        corr_tag = cfg["corr_tag"]
        if cfg["no_multi_modal"] is True:
            assert corr_tag == "no-correction", cfg["corr_tag"]
            tag = "UM"
        elif corr_tag == "no-correction":
            tag = "BASE"
        elif corr_tag.startswith("corrector_7-gwot-allpix-patches/"):
            if corr_tag.endswith("_rndpaired"):
                tag = "RND"
            else:
                tag = "FGW"
        elif corr_tag.startswith("corrector_z-score-corr/"):
            tag = "NORM"
        else:
            raise ValueError(f"{corr_tag=} not supported ({slurm_logs_f[k]=})")

        # search accuracy
        marker = "top1,top3,top5,rank-q.95"
        if "oom_kill" in log:
            print(f"error in {slurm_logs_f[k]}")
            acc1 = acc3 = acc5 = float("nan")
        elif "+ res=1" in lines:
            print(f"error in {slurm_logs_f[k]}")
            acc1 = acc3 = acc5 = float("nan")
        elif marker not in lines:
            print(f"running: {slurm_logs_f[k]}")
            acc1 = acc3 = acc5 = float("nan")
        else:
            acc_line = lines[lines.index(marker) + 1]
            acc1, acc3, acc5, _ = map(float, acc_line.split(","))

        # search time per query
        marker = "===== Timing ====="
        qtime = float(lines[lines.index(marker) + 1].split()[2])

        # total memory and total wall time for the experiment
        marker = "Resources Used"
        if marker in lines:
            mem = mem_to_gb(lines[lines.index(marker) + 2].split()[-1])
            if mem < 1e-1:
                # sometimes slurms reports an error wrt the total memory used
                mem = float("nan")
            twtime = time_to_seconds(lines[lines.index(marker) + 4].split()[-1])
        else:
            mem = twtime = float("nan")

        ts = float(cfg["tile_size_um"])
        rows.append((ts, cfg["model"], tag, 100 * acc1, 100 * acc3, 100 * acc5, qtime, mem, twtime))

    return pd.DataFrame(rows, columns=columns)  # type: ignore


parent_directory = pathlib.Path(".runexp/search")
assert parent_directory.is_dir()
configurations = [gather(p) for p in parent_directory.iterdir()]

# merge all configurations
complete_df = pd.concat(configurations)
complete_df.sort_values(["tile_size_um", "model", "tag"], inplace=True)
complete_df.reset_index(drop=True, inplace=True)

# complete_df["qtime"] = complete_df["qtime"].map("{:,.1e}s".format)
# complete_df["mem"] = complete_df["mem"].map("{:,.2f}GiB".format)
# complete_df["twtime"] = complete_df["twtime"].map("{:,.1e}s".format)

# print(complete_df.to_csv())


def build_metric_table(df_, metric_):

    tab = (
        df_
        .pivot_table(
            index="tag",
            columns=["tile_size_um", "model"],
            values=metric_,
            aggfunc="first"   # safe: already aggregated per config
        )
        .reindex(tags)
    )

    col_order = pd.MultiIndex.from_product([tiles, models])
    tab = tab.reindex(columns=col_order)

    return tab


df = complete_df.copy()
tiles  = sorted(df["tile_size_um"].unique())
models = sorted(df["model"].unique())
tags   = list(df["tag"].unique())   # ONLY existing tags

top1_tab = build_metric_table(df, "top1")
top5_tab = build_metric_table(df, "top5")
qtime_tab = build_metric_table(df, "qtime")
mem_tab   = build_metric_table(df, "mem")


def latex_header(tiles, models):

    h1 = "&& "
    h2 = "&& "

    for t in tiles:
        h1 += f"\\multicolumn{{{len(models)}}}{{c|}}{{{int(t)}µm}} & "

    for _ in tiles:
        for m in models:
            h2 += f"{m} & "

    return (
        h1.rstrip("& ") + " \\\\\n" +
        h2.rstrip("& ") + " \\\\"
    )


def latex_block(table, title, numeric=True):

    rows = []
    n_tags = len(table.index)

    for i, tag in enumerate(table.index):

        values = []

        for v in table.loc[tag]:
            if pd.isna(v):
                values.append("--")
            else:
                if numeric:
                    values.append(f"{v:.2f}")
                else:
                    values.append(str(v))

        prefix = (
            f"\\multirow{{{n_tags}}}{{*}}{{\\rotatebox{{90}}{{\\textbf{{{title}}}}}}} & {tag}"
            if i == 0 else
            f"& {tag}"
        )

        rows.append(prefix + " & " + " & ".join(values) + " \\\\")

    return "\n".join(rows)


header = latex_header(tiles, models)

top1_body = latex_block(top1_tab, "Top-1", numeric=True)
top5_body = latex_block(top5_tab, "Top-5", numeric=True)
qt_body   = latex_block(qtime_tab, "Qtime", numeric=False)
mem_body  = latex_block(mem_tab, "Mem", numeric=False)

ncols = len(tiles) * len(models)

print(f"\\begin{{tabular}}{{cl||{'c'*ncols}}}")
print(header)
print("\\hline\\hline")
print(top1_body)
print("\\hline")
print(top5_body)
print("\\hline")
print(qt_body)
print("\\hline")
print(mem_body)
print("\\end{tabular}")
