import argparse
import fmodel
import json
import numpy as np

parser = argparse.ArgumentParser(description='Predict using the Image Classifier CLI App')

# Basic usage: python predict.py /path/to/image checkpoint
# Options: 
#     * Return top K most likely classes: python predict.py input checkpoint --top_k 3 
#     * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json 
#     * Use GPU for inference: python predict.py input checkpoint --gpu

parser.add_argument('image_path', type=str)
parser.add_argument('checkpoint_path', type=str)
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--category_names', type=str, default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()
img_path = args.image_path
k_number = args.top_k
use_gpu = args.gpu
checkpoint_path = args.checkpoint_path
category_names = args.category_names

def main():
    model = fmodel.load_checkpoint(checkpoint_path)
    
    print('************ Prediction started! **************\n')
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    probabilities = fmodel.predict(img_path, model, k_number, use_gpu)
    labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    probability = np.array(probabilities[0][0])
    
    for i in range(k_number):
        print(f"{labels[i]} with a probability {probability[i]}")
        
    print('\n************ Prediction finished! **************')
    
if __name__== "__main__":
    main()