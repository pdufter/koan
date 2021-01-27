./build/koan -V 2000000 \
             --epochs 10 \
             --dim 300 \
             --negatives 5 \
             --context-size 5 \
             -l 0.075 \
             --threads 16 \
             --cbow true \
             --min-count 2 \
             --file /Users/philipp/Downloads/corpus.txt

python ~/Dropbox/PhD/tools/projector/vis_embed_online.py /Users/Philipp/Dropbox/PhD/projects/koan/embeddings_2021-01-27_17:10:33.txt