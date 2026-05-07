import kagglehub

# Download latest version
path = kagglehub.competition_download('playground-series-s6e2')

print("Path to competition files:", path)