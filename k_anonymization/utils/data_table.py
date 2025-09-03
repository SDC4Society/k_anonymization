from IPython.display import HTML, display
from itables import init_notebook_mode
from itables import show as itables_show
from pandas.core.frame import DataFrame

css = """
.jp-RenderedJSON .filter {
  display: none;
}
.dt-scroll {
  margin: 0 0 !important;
}
.dtsp-panesContainer {
  width: 100% !important;
}
.dt-scroll-body {
  height: auto !important;
}

.dt-container caption{
  caption-side: top;
  font-size: large;
  font-weight: bold;
}

.dt-info a:after {
  content:"\\A";
  white-space: pre;
}
"""
display(HTML(f"<style>{css}</style>" ""))

init_notebook_mode(all_interactive=True)


def show(
    data: DataFrame,
    table_name: str = None,
    search_columns_per_row: int = 5,
    max_bytes: str = "64KB",
):
    _search_panes_props = {"features": "searchPanes"}

    if table_name:
        _table_name = table_name.replace(" ", "_").lower()
        table_css = f"""
            #{_table_name}-search-panes div.dtsp-searchPanes {"{"}
                width: 100% !important;
                display: grid !important;
                column-gap: 1% !important;
                grid-template-columns: repeat(auto-fit, minmax({int(100/search_columns_per_row)-1}%, {int(100/search_columns_per_row)-1}%)) !important;
                justify-content: unset !important;
            {"}"}
            #{_table_name}-search-panes div.dtsp-columns-{search_columns_per_row} {"{"}
                max-width: unset !important;
                min-width: unset !important;
            {"}"}
        """
        display(HTML(f"<style>{table_css}</style>" ""))
        _search_panes_props["id"] = f"{_table_name}-search-panes"

    itables_show(
        data,
        table_name,
        maxBytes=max_bytes,
        layout={
            "top1": _search_panes_props,
            "topStart": "searchBuilder",
            "topEnd": "pageLength",
            "bottomEnd": "paging",
        },
        searchPanes={
            "threshold": 1,
            "columns": [i for i in range(1, len(data.keys()))],
            "layout": f"columns-{search_columns_per_row}",
            "cascadePanes": True,
            "initCollapsed": True,
            "dtOpts": {"order": [[1, "desc"], [0, "asc"]]},
        },
    )
