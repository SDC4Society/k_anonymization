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
    _table_name = (
        table_name.replace(" ", "_").lower() if table_name else "custom-itables"
    )
    table_css = f"""
        #{_table_name}-buttons .dt-button-collection {"{"}
            padding: 10px
        {"}"}
        #{_table_name}-buttons div.dtsp-searchPanes {"{"}
            width: 100% !important;
            column-gap: 10px !important;
            justify-content: flex-start !important;
        {"}"}
    """
    display(HTML(f"<style>{table_css}</style>" ""))

    itables_show(
        data,
        table_name,
        maxBytes=max_bytes,
        layout={
            "top": {
                "id": f"{_table_name}-buttons",
                "features": [
                    {
                        "buttons": [
                            "pageLength",
                            {
                                "extend": "searchPanes",
                                "config": {
                                    "threshold": 1,
                                    "columns": [i for i in range(1, len(data.keys()))],
                                    "layout": f"columns-{search_columns_per_row}",
                                    "cascadePanes": True,
                                    "initCollapsed": True,
                                    "dtOpts": {"order": [[1, "desc"], [0, "asc"]]},
                                },
                            },
                            "searchBuilder",
                        ]
                    }
                ],
            },
            "topStart": None,
            "topEnd": None,
            "bottomStart": None,
            "bottomEnd": None,
            "bottom1": ["paging"],
            "bottom2": ["info"],
        },
        language={
            "searchPanes": {
                "collapse": {0: "Filters", "_": "Filters (%d)"},
            }
        },
    )
