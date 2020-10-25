using AdaptiveSparseGrids
using Documenter

makedocs(;
    modules=[AdaptiveSparseGrids],
    authors="Jacob Adenbaum",
    repo="https://github.com/jacobadenbaum/PanelLag.jl/blob/{commit}{path}#L{line}",
    sitename="AdaptiveSparseGrids.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jacobadenbaum.github.io/AdaptiveSparseGrids.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jacobadenbaum/AdaptiveSparseGrids.jl",
)
