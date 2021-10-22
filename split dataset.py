import splitfolders

input_folder = 'original dataset'
output = 'dataset'

splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.8, .2))
