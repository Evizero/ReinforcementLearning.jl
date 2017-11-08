using Documenter, ReinforcementLearning

makedocs(
    modules = [ReinforcementLearning],
    clean = false,
    format = :html,
#    assets = [
#        joinpath("assets", "favicon.ico"),
#        joinpath("assets", "style.css"),
#    ],
    sitename = "Reinforcement Learning",
    authors = "Christof Stocker",
    linkcheck = !("skiplinks" in ARGS),
    pages = Any[
        "Home" => "index.md",
        "Developer Documentation" => Any[
            joinpath("devdocs", "design.md"),
        ],
        hide("Indices" => "indices.md"),
        "LICENSE.md",
    ],
    html_prettyurls = !("local" in ARGS),
)

deploydocs(
    repo = "github.com/Evizero/ReinforcementLearning.jl.git",
    target = "build",
    julia = "0.6",
    deps = nothing,
    make = nothing,
)
