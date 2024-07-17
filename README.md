# This folder contains the following
- This README, which contains:
	- This overview of files in the project folder.
	- A comparison between my replicated results and the results of the paper.
	- A discussion of my MultiTree results.
	- A future research suggestion.
- The original paper on which this project was based. This is the file ThePaper.pdf.
- My main results. Namely, replicating the results of the paper.
	- My implementation can be found in the file main.hs. 
		- To compile this file, criterion and deepseq should be installed and added to the environment file. This can be achieved by running "cabal install --lib criterion" and "cabal install --lib deepseq".
	   	- This file should be compiled with the O flag; ghc -O main.hs. Otherwise, the results will not be duplicated as the defined Haskell optimization rules will not be used.
	- My results can be found in the file ReplicatedResults.pdf.
	- I compare my results to the results of the paper after this section of this README file.
- My first additional assignment in which I implemented a Church and CoChurch encoding for MultiTrees.
   MultiTrees here are trees with any number of children, and values stored at the leaves.
	- The implementation can be found in the file multitree.hs.
		- To compile this file, criterion and deepseq should again be installed and added to the environment file.
		- This file should be compiled with the O flag; ghc -O multitree.hs. Otherwise, the results will not be duplicated as the defined Haskell optimization rules will not be used.
	- My results can be found in the file ResultsMultiTree.pdf.
	- I discuss these results at the end of this README file.
- My second additional assignment in which I implemented a Church encoding for lists.
	- The implementation can be found in the file list.hs.
		- To compile this file, containers should be installed and added to the environment file. This can be achieved by running "cabal install --lib containers".
  	- An explanation of this implementation can be found in the file ListChurchEncodingExplanation.pdf. I heavily recommend going over the implementation and this explanation file together. I further explain this is the explanation file.


## Comparison between my replicated results and the results of the paper
I have mostly been able to replicate the results of the paper. 
I have run the results using Windows 11 on an AMD Ryzen 9 7900X 12-Core Processor of 4.7 GHz, with a 32GB RAM and a 64MB L3 cache. I used GHC version 9.4.8.
The single function benchmarks are very similar, showing the overhead of the encodings for singular functions, especially for append, which corresponds with the paper.
The small pipeline benchmarks are also very similar. Showing that the CoChurch encoding is often very efficient even for small pipelines, which corresponds with the paper.
The most important replicated results are the large pipelines, which show that the Church and CoChurch encoding are both faster than the normal pipelines, which corresponds with the paper.
The replicated results of the large append pipeline do also show the overhead Church and CoChurch encodings pose when unfused.
Lastly, the time it takes for my functions to run is significantly faster than in the paper. I think this is due to the Haskell compiler having improved, as the paper is from 2011, 
and my setup being more powerful than the one that was used by the paper. It can also have to do with the benchmarking tool I used, namely Criterion, as I have no clue what tool the paper used.


## Discussion of my MultiTree results
I have run the results using Windows 11 on an AMD Ryzen 9 7900X 12-Core Processor of 4.7 GHz, with a 32GB RAM and a 64MB L3 cache. I used GHC version 9.4.8.
The single function results are very similar to the single function results belonging to my replicated results.
An interesting observation is that the Church-encoded small pipelines often perform closely to and sometimes better than the normal variants and CoChurch-encoded variants.
This is different from Church-encoded small pipeline results of the replicated results, as there these Church-encoded pipelines perform more often (a lot) worse.
Again, however, the CoChurch encoding is often very efficient even for small pipelines, which corresponds with the paper and replicated results.
The large pipeline benchmarks again show the efficiency of Church and CoChurch encodings, as in the pipeline benchmark, both the Church and CoChurch encoding perform better, and in the append pipeline benchmark, the Church encoding performs better.
Also, the results of the large append pipeline do show the overhead Church and CoChurch encodings pose when unfused.

## future research idea
I implemented the MultiTree structure using standard Haskell lists. It could be interesting to implement them using a (Co)Church-encoded variant of lists.
This would create a structure that is (Co)Church encoded, and is made using another (Co)Church-encoded structure.
I tried this out, but the limitation of the map function I discuss in ListChurchEncodingExplanation.pdf makes this impossible, as the functions over MultiTrees require a lot of mapping.
