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


nice -n 19 ./build/koan -V 20000 \
             --epochs 10 \
             --dim 20 \
             --hidden 300 \
             --negatives 5 \
             --context-size 5 \
             -l 0.075 \
             --threads 40 \
             --cbow true \
             --min-count 5 \
             --file ../pytorch-sgns/data/corpus.txt
python ~/Dropbox/PhD/tools/projector/vis_embed_online.py /Users/philipp/Downloads/embeddings_2021-01-27_18:35:08.txt
